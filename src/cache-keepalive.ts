export const KEEPALIVE_CACHE_TTL_MS = 60 * 60 * 1000;
export const KEEPALIVE_DEFAULT_DELAY_MS = 55 * 60 * 1000;
export const KEEPALIVE_DEFAULT_MAX_PINGS = 6;
export const KEEPALIVE_MAX_PINGS = 24;
export const KEEPALIVE_MIN_PROMPT_TOKENS = 10_000;

export type KeepalivePolicy = {
  enabled: boolean;
  delayMs: number;
  maxPings: number;
  ttlMs: number;
  minPromptTokens: number;
};

export type KeepalivePingResult = {
  cacheRead: number;
  cacheWrite: number;
};

type TimerHandle = ReturnType<typeof setTimeout>;

export type KeepaliveRuntime<T> = {
  now: () => number;
  setTimer: (callback: () => void, delayMs: number) => TimerHandle;
  clearTimer: (timer: TimerHandle) => void;
  ping: (payload: T, signal: AbortSignal) => Promise<KeepalivePingResult>;
  debug?: (message: string) => void;
};

type KeepaliveState<T> = {
  payload: T;
  policy: KeepalivePolicy;
  count: number;
  lastAccessAt: number;
  timer?: TimerHandle;
  controller?: AbortController;
};

function parseMaxPings(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === "") return KEEPALIVE_DEFAULT_MAX_PINGS;
  const value = Math.floor(Number(raw));
  return Number.isSafeInteger(value) && value >= 0 && value <= KEEPALIVE_MAX_PINGS
    ? value
    : KEEPALIVE_DEFAULT_MAX_PINGS;
}

function parseDelayMs(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === "") return KEEPALIVE_DEFAULT_DELAY_MS;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 && value < KEEPALIVE_CACHE_TTL_MS
    ? value
    : KEEPALIVE_DEFAULT_DELAY_MS;
}

export function resolveKeepalivePolicy(
  env: NodeJS.ProcessEnv = process.env,
): KeepalivePolicy {
  return {
    enabled: env.PI_CACHE_RETENTION === "long",
    delayMs: parseDelayMs(env.PI_CACHE_KEEPALIVE_DELAY_MS),
    maxPings: parseMaxPings(env.PI_CACHE_KEEPALIVE),
    ttlMs: KEEPALIVE_CACHE_TTL_MS,
    minPromptTokens: KEEPALIVE_MIN_PROMPT_TOKENS,
  };
}

/**
 * Owns the one keepalive chain for a Pi process.
 *
 * Each real request starts a new generation. Only that request may publish the
 * exact replay payload after it succeeds, so an older concurrent completion can
 * never replace a newer request's cache state.
 */
export class CacheKeepaliveManager<T> {
  private generation = 0;
  private state: KeepaliveState<T> | undefined;

  constructor(private readonly runtime: KeepaliveRuntime<T>) {}

  beginRequest(): number {
    this.generation += 1;
    this.clearState();
    return this.generation;
  }

  finishRequest(
    generation: number,
    payload: T,
    promptTokens: number,
    requestStartedAt: number,
    policy: KeepalivePolicy,
  ): void {
    if (generation !== this.generation) return;
    this.clearState();
    if (!policy.enabled || policy.maxPings === 0) return;
    if (!Number.isFinite(promptTokens) || promptTokens < policy.minPromptTokens) return;

    const ageMs = Math.max(0, this.runtime.now() - requestStartedAt);
    if (ageMs >= policy.ttlMs) {
      this.runtime.debug?.("skip: cache TTL expired before response completed");
      return;
    }

    const state: KeepaliveState<T> = {
      payload,
      policy,
      count: 0,
      lastAccessAt: requestStartedAt,
    };
    this.state = state;
    this.arm(state);
    this.runtime.debug?.(
      `scheduled promptTokens=${promptTokens} maxPings=${policy.maxPings}`,
    );
  }

  cancel(): void {
    this.generation += 1;
    this.clearState();
  }

  private clearState(): void {
    const state = this.state;
    this.state = undefined;
    if (!state) return;
    if (state.timer) this.runtime.clearTimer(state.timer);
    state.controller?.abort();
  }

  private stop(state: KeepaliveState<T>): void {
    if (this.state === state) this.clearState();
  }

  private arm(state: KeepaliveState<T>): void {
    if (this.state !== state) return;
    const ageMs = Math.max(0, this.runtime.now() - state.lastAccessAt);
    if (ageMs >= state.policy.ttlMs) {
      this.runtime.debug?.("skip: cache TTL already expired");
      this.stop(state);
      return;
    }

    const remainingMs = Math.max(0, state.policy.delayMs - ageMs);
    const timer = this.runtime.setTimer(() => {
      if (this.state === state) state.timer = undefined;
      void this.runPing(state);
    }, remainingMs);
    timer.unref?.();
    state.timer = timer;
  }

  private async runPing(state: KeepaliveState<T>): Promise<void> {
    if (this.state !== state || state.controller) return;

    const pingStartedAt = this.runtime.now();
    if (pingStartedAt - state.lastAccessAt >= state.policy.ttlMs) {
      this.runtime.debug?.("skip: cache TTL already expired");
      this.stop(state);
      return;
    }

    const controller = new AbortController();
    state.controller = controller;
    let result: KeepalivePingResult;
    try {
      result = await this.runtime.ping(state.payload, controller.signal);
    } catch (error) {
      if (this.state !== state) return;
      state.controller = undefined;
      this.runtime.debug?.(
        `ping failed: ${error instanceof Error ? error.name : typeof error}`,
      );
      this.stop(state);
      return;
    } finally {
      // `message_start` is enough to establish the cache result. Ensure a
      // provider stream cannot continue generating output after ping() returns.
      controller.abort();
    }

    if (this.state !== state) return;
    state.controller = undefined;
    this.runtime.debug?.(
      `ping result cacheRead=${result.cacheRead} cacheWrite=${result.cacheWrite}`,
    );

    if (!Number.isFinite(result.cacheRead) || result.cacheRead <= 0) {
      this.runtime.debug?.("stop: keepalive did not read from cache");
      this.stop(state);
      return;
    }
    if (this.runtime.now() - pingStartedAt >= state.policy.ttlMs) {
      this.runtime.debug?.("stop: ping response arrived after refreshed TTL expired");
      this.stop(state);
      return;
    }

    state.count += 1;
    state.lastAccessAt = pingStartedAt;
    if (state.count >= state.policy.maxPings) {
      this.runtime.debug?.(`done: reached maxPings=${state.policy.maxPings}`);
      this.stop(state);
      return;
    }
    this.arm(state);
  }
}
