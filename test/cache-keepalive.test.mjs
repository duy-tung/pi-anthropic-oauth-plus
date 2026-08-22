import assert from "node:assert/strict";
import test from "node:test";
import {
  CacheKeepaliveManager,
  KEEPALIVE_CACHE_TTL_MS,
  KEEPALIVE_DEFAULT_DELAY_MS,
  KEEPALIVE_DEFAULT_MAX_PINGS,
  KEEPALIVE_MAX_PINGS,
  KEEPALIVE_MIN_PROMPT_TOKENS,
  resolveKeepalivePolicy,
} from "../.test-dist/src/cache-keepalive.js";

const minute = 60_000;
const settle = () => new Promise((resolve) => setImmediate(resolve));

class FakeRuntime {
  nowMs = 0;
  timers = [];
  pingCalls = [];
  pingImpl = async () => ({ cacheRead: 100, cacheWrite: 0 });

  runtime() {
    return {
      now: () => this.nowMs,
      setTimer: (callback, delayMs) => {
        const timer = {
          callback,
          delayMs,
          at: this.nowMs + delayMs,
          cleared: false,
          fired: false,
          unrefCalled: false,
          unref() {
            this.unrefCalled = true;
          },
        };
        this.timers.push(timer);
        return timer;
      },
      clearTimer: (timer) => {
        timer.cleared = true;
      },
      ping: async (payload, signal) => {
        this.pingCalls.push({ payload, signal, at: this.nowMs });
        return this.pingImpl(payload, signal);
      },
    };
  }

  activeTimers() {
    return this.timers.filter((timer) => !timer.cleared && !timer.fired);
  }

  async fire(timer, at = timer.at) {
    this.nowMs = at;
    timer.fired = true;
    timer.callback();
    await settle();
  }
}

function enabledPolicy(overrides = {}) {
  return {
    enabled: true,
    delayMs: KEEPALIVE_DEFAULT_DELAY_MS,
    maxPings: KEEPALIVE_DEFAULT_MAX_PINGS,
    ttlMs: KEEPALIVE_CACHE_TTL_MS,
    minPromptTokens: KEEPALIVE_MIN_PROMPT_TOKENS,
    ...overrides,
  };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test("policy defaults to six pings and validates overrides", () => {
  const defaults = resolveKeepalivePolicy({ PI_CACHE_RETENTION: "long" });
  assert.equal(defaults.enabled, true);
  assert.equal(defaults.maxPings, 6);
  assert.equal(defaults.delayMs, 55 * minute);

  assert.equal(
    resolveKeepalivePolicy({ PI_CACHE_RETENTION: "long", PI_CACHE_KEEPALIVE: "0" }).maxPings,
    0,
  );
  assert.equal(
    resolveKeepalivePolicy({ PI_CACHE_RETENTION: "long", PI_CACHE_KEEPALIVE: "8.9" }).maxPings,
    8,
  );
  assert.equal(
    resolveKeepalivePolicy({ PI_CACHE_RETENTION: "long", PI_CACHE_KEEPALIVE: "bad" }).maxPings,
    6,
  );
  assert.equal(
    resolveKeepalivePolicy({
      PI_CACHE_RETENTION: "long",
      PI_CACHE_KEEPALIVE: String(KEEPALIVE_MAX_PINGS),
    }).maxPings,
    KEEPALIVE_MAX_PINGS,
  );
  for (const unsafe of ["25", "1e100", String(Number.MAX_SAFE_INTEGER), "Infinity"]) {
    assert.equal(
      resolveKeepalivePolicy({ PI_CACHE_RETENTION: "long", PI_CACHE_KEEPALIVE: unsafe }).maxPings,
      6,
      `unsafe max pings should fall back: ${unsafe}`,
    );
  }
  assert.equal(
    resolveKeepalivePolicy({
      PI_CACHE_RETENTION: "long",
      PI_CACHE_KEEPALIVE_DELAY_MS: String(KEEPALIVE_CACHE_TTL_MS),
    }).delayMs,
    55 * minute,
  );
  assert.equal(resolveKeepalivePolicy({}).enabled, false);
});

test("first ping is anchored to real provider request start", () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());
  const generation = manager.beginRequest();

  fake.nowMs = 10 * minute;
  manager.finishRequest(generation, { id: "request" }, 20_000, 0, enabledPolicy());

  const [timer] = fake.activeTimers();
  assert.ok(timer);
  assert.equal(timer.delayMs, 45 * minute);
  assert.equal(timer.at, 55 * minute);
  assert.equal(timer.unrefCalled, true);
});

test("late real completion is immediate before TTL and skipped at TTL", () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());

  let generation = manager.beginRequest();
  fake.nowMs = 56 * minute;
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
  assert.equal(fake.activeTimers()[0]?.delayMs, 0);

  generation = manager.beginRequest();
  fake.nowMs = 60 * minute;
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
  assert.equal(fake.activeTimers().length, 0);
});

test("six confirmed cache reads run and no seventh ping is armed", async () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());
  const payload = { exact: true };
  const generation = manager.beginRequest();
  manager.finishRequest(generation, payload, 20_000, 0, enabledPolicy());

  for (let i = 1; i <= 6; i += 1) {
    const [timer] = fake.activeTimers();
    assert.ok(timer, `missing timer for ping ${i}`);
    await fake.fire(timer);
    assert.equal(fake.pingCalls.length, i);
    assert.strictEqual(fake.pingCalls.at(-1).payload, payload);
    assert.equal(fake.pingCalls.at(-1).signal.aborted, true);
  }
  assert.equal(fake.activeTimers().length, 0);
});

test("positive ping rearms from ping start, not response completion", async () => {
  const fake = new FakeRuntime();
  const pending = deferred();
  fake.pingImpl = () => pending.promise;
  const manager = new CacheKeepaliveManager(fake.runtime());
  const generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());

  const [first] = fake.activeTimers();
  await fake.fire(first, 55 * minute);
  fake.nowMs = 56 * minute;
  pending.resolve({ cacheRead: 100, cacheWrite: 0 });
  await settle();

  const [next] = fake.activeTimers();
  assert.ok(next);
  assert.equal(next.at, 110 * minute);
  assert.equal(next.delayMs, 54 * minute);
});

test("late positive ping completion does not rearm past its TTL", async () => {
  const fake = new FakeRuntime();
  const pending = deferred();
  fake.pingImpl = () => pending.promise;
  const manager = new CacheKeepaliveManager(fake.runtime());
  const generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());

  await fake.fire(fake.activeTimers()[0], 55 * minute);
  fake.nowMs = 115 * minute;
  pending.resolve({ cacheRead: 100, cacheWrite: 0 });
  await settle();
  assert.equal(fake.activeTimers().length, 0);
});

test("zero cache read, provider error, and delayed timer fail closed", async () => {
  {
    const fake = new FakeRuntime();
    fake.pingImpl = async () => ({ cacheRead: 0, cacheWrite: 20_000 });
    const manager = new CacheKeepaliveManager(fake.runtime());
    const generation = manager.beginRequest();
    manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
    await fake.fire(fake.activeTimers()[0]);
    assert.equal(fake.pingCalls.length, 1);
    assert.equal(fake.activeTimers().length, 0);
  }

  {
    const fake = new FakeRuntime();
    fake.pingImpl = async () => {
      throw new Error("provider down");
    };
    const manager = new CacheKeepaliveManager(fake.runtime());
    const generation = manager.beginRequest();
    manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
    await fake.fire(fake.activeTimers()[0]);
    assert.equal(fake.activeTimers().length, 0);
  }

  {
    const fake = new FakeRuntime();
    const manager = new CacheKeepaliveManager(fake.runtime());
    const generation = manager.beginRequest();
    manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
    await fake.fire(fake.activeTimers()[0], 60 * minute);
    assert.equal(fake.pingCalls.length, 0);
    assert.equal(fake.activeTimers().length, 0);
  }
});

test("non-finite cache reads cannot rearm", async () => {
  for (const cacheRead of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    const fake = new FakeRuntime();
    fake.pingImpl = async () => ({ cacheRead, cacheWrite: 0 });
    const manager = new CacheKeepaliveManager(fake.runtime());
    const generation = manager.beginRequest();
    manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
    await fake.fire(fake.activeTimers()[0]);
    assert.equal(fake.activeTimers().length, 0);
  }
});

test("disabled and small prompts never arm", () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());

  let generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy({ enabled: false }));
  assert.equal(fake.activeTimers().length, 0);

  generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 9_999, 0, enabledPolicy());
  assert.equal(fake.activeTimers().length, 0);

  generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy({ maxPings: 0 }));
  assert.equal(fake.activeTimers().length, 0);
});

test("older real completion cannot replace a newer published state", () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());
  const oldGeneration = manager.beginRequest();
  const newGeneration = manager.beginRequest();

  manager.finishRequest(newGeneration, { id: "new" }, 20_000, 0, enabledPolicy());
  const newTimer = fake.activeTimers()[0];
  manager.finishRequest(oldGeneration, { id: "old" }, 20_000, 0, enabledPolicy());

  assert.deepEqual(fake.activeTimers(), [newTimer]);
});

test("new real request clears timer and stale completion cannot publish", async () => {
  const fake = new FakeRuntime();
  const manager = new CacheKeepaliveManager(fake.runtime());
  const oldGeneration = manager.beginRequest();
  manager.finishRequest(oldGeneration, { id: "old" }, 20_000, 0, enabledPolicy());
  const staleTimer = fake.activeTimers()[0];

  fake.nowMs = 1 * minute;
  const newGeneration = manager.beginRequest();
  assert.equal(staleTimer.cleared, true);
  manager.finishRequest(oldGeneration, { id: "stale-finish" }, 20_000, 0, enabledPolicy());
  manager.finishRequest(newGeneration, { id: "new" }, 20_000, fake.nowMs, enabledPolicy());

  staleTimer.callback();
  await settle();
  assert.equal(fake.pingCalls.length, 0);
  assert.equal(fake.activeTimers().length, 1);
  assert.equal(fake.activeTimers()[0].at, 56 * minute);
});

test("new request aborts in-flight ping and stale success cannot rearm", async () => {
  const fake = new FakeRuntime();
  const pending = deferred();
  fake.pingImpl = () => pending.promise;
  const manager = new CacheKeepaliveManager(fake.runtime());
  const oldGeneration = manager.beginRequest();
  manager.finishRequest(oldGeneration, { id: "old" }, 20_000, 0, enabledPolicy());
  await fake.fire(fake.activeTimers()[0], 55 * minute);
  const oldSignal = fake.pingCalls[0].signal;
  assert.equal(oldSignal.aborted, false);

  const newGeneration = manager.beginRequest();
  assert.equal(oldSignal.aborted, true);
  manager.finishRequest(newGeneration, { id: "new" }, 20_000, fake.nowMs, enabledPolicy());
  const newTimer = fake.activeTimers()[0];

  pending.resolve({ cacheRead: 100, cacheWrite: 0 });
  await settle();
  assert.deepEqual(fake.activeTimers(), [newTimer]);
});

test("stale ping error cannot cancel a newer state", async () => {
  const fake = new FakeRuntime();
  const pending = deferred();
  fake.pingImpl = () => pending.promise;
  const manager = new CacheKeepaliveManager(fake.runtime());
  const oldGeneration = manager.beginRequest();
  manager.finishRequest(oldGeneration, { id: "old" }, 20_000, 0, enabledPolicy());
  await fake.fire(fake.activeTimers()[0], 55 * minute);

  const newGeneration = manager.beginRequest();
  manager.finishRequest(newGeneration, { id: "new" }, 20_000, fake.nowMs, enabledPolicy());
  const newTimer = fake.activeTimers()[0];

  pending.reject(new Error("stale failure"));
  await settle();
  assert.deepEqual(fake.activeTimers(), [newTimer]);
});

test("shutdown clears pending and in-flight work", async () => {
  const fake = new FakeRuntime();
  const pending = deferred();
  fake.pingImpl = () => pending.promise;
  const manager = new CacheKeepaliveManager(fake.runtime());
  const generation = manager.beginRequest();
  manager.finishRequest(generation, {}, 20_000, 0, enabledPolicy());
  await fake.fire(fake.activeTimers()[0], 55 * minute);
  const signal = fake.pingCalls[0].signal;

  manager.cancel();
  assert.equal(signal.aborted, true);
  pending.resolve({ cacheRead: 100, cacheWrite: 0 });
  await settle();
  assert.equal(fake.activeTimers().length, 0);
});
