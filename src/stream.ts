import {
  closeSync,
  constants as fsConstants,
  fchmodSync,
  fstatSync,
  mkdirSync,
  openSync,
  writeSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/messages.js";
import {
  type AnthropicAllowedFallbackModel,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  calculateCost,
  type Context,
  createAssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
  type StopReason,
} from "@earendil-works/pi-ai";
import { isClaudeOAuthAccessToken, USER_AGENT } from "./auth.js";
import {
  convertPiMessagesToAnthropic,
  convertPiToolsToAnthropic,
  fromClaudeCodeToolName,
  type IndexedBlock,
} from "./convert.js";
import { buildAnthropicSystemPrompt } from "./prompt.js";
import {
  CacheKeepaliveManager,
  type KeepalivePingResult,
  resolveKeepalivePolicy,
} from "./cache-keepalive.js";

const FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14";
const SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01";
const REQUIRED_BETAS = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  "interleaved-thinking-2025-05-14",
] as const;

// Fine-grained tool streaming ships raw, unvalidated tool-input JSON. Large
// edits containing quotes/newlines can arrive with a field swallowing the rest
// of the structure. Always keep the server-buffered default, even if another
// extension asks for the beta while contributing unrelated headers.
const BLOCKED_BETAS = new Set([FINE_GRAINED_TOOL_STREAMING_BETA]);

export type AnthropicRequestParams = MessageCreateParamsStreaming & {
  fallbacks?: Array<{ model: string }>;
  speed?: "fast";
};

// ---------------------------------------------------------------------------
// Cache keepalive: with PI_CACHE_RETENTION=long (1h TTL), replay the exact last
// successful request at 55-minute intervals and stop after `message_start`.
// The manager anchors TTL accounting to request start, serializes generations,
// aborts stale/in-flight pings, and only rearms after a confirmed cache read.
// ---------------------------------------------------------------------------

type AnthropicKeepalivePayload = {
  client: Anthropic;
  params: AnthropicRequestParams;
};

const KEEPALIVE_DEBUG_MAX_BYTES = 64 * 1024;

export function keepaliveDebug(message: string): void {
  if (process.env.PI_CACHE_KEEPALIVE_DEBUG !== "1") return;
  let fd: number | undefined;
  try {
    const configDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
    const cacheDir = join(configDir, "cache");
    mkdirSync(cacheDir, { recursive: true, mode: 0o700 });
    const file = join(cacheDir, "cache-keepalive.log");
    fd = openSync(
      file,
      fsConstants.O_WRONLY |
        fsConstants.O_APPEND |
        fsConstants.O_CREAT |
        fsConstants.O_NOFOLLOW,
      0o600,
    );
    const stat = fstatSync(fd);
    if (!stat.isFile()) return;
    fchmodSync(fd, 0o600);
    const clean = message.replace(/[\r\n]+/g, " ").slice(0, 512);
    const line = `${new Date().toISOString()} ${clean}\n`;
    if (stat.size + Buffer.byteLength(line) > KEEPALIVE_DEBUG_MAX_BYTES) return;
    writeSync(fd, line);
  } catch {
    // Debugging must never affect the provider request path.
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {}
    }
  }
}

export async function pingAnthropicCache(
  payload: AnthropicKeepalivePayload,
  signal: AbortSignal,
): Promise<KeepalivePingResult> {
  const stream = await payload.client.messages.create(payload.params, { signal });
  for await (const event of stream) {
    if (event.type !== "message_start") continue;
    const usage = event.message.usage as unknown as {
      cache_read_input_tokens?: number;
      cache_creation_input_tokens?: number;
    };
    return {
      cacheRead: usage?.cache_read_input_tokens ?? 0,
      cacheWrite: usage?.cache_creation_input_tokens ?? 0,
    };
  }
  return { cacheRead: 0, cacheWrite: 0 };
}

const cacheKeepalive = new CacheKeepaliveManager<AnthropicKeepalivePayload>({
  now: Date.now,
  setTimer: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimer: (timer) => clearTimeout(timer),
  ping: pingAnthropicCache,
  debug: keepaliveDebug,
});

export function cancelCacheKeepalive(): void {
  cacheKeepalive.cancel();
}

function mapStopReason(reason: string | null | undefined): StopReason {
  switch (reason) {
    case "end_turn":
    case "pause_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "toolUse";
    default:
      return "error";
  }
}

function headersToRecord(headers: Headers): Record<string, string> {
  const record: Record<string, string> = {};
  headers.forEach((value, key) => {
    record[key] = value;
  });
  return record;
}

export function readCacheWrite1h(usage: unknown): number | undefined {
  const value = (usage as { cache_creation?: { ephemeral_1h_input_tokens?: unknown } })
    ?.cache_creation?.ephemeral_1h_input_tokens;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

type AnthropicUsageFields = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
  output_tokens_details?: { thinking_tokens?: number | null };
};

export function applyAnthropicUsage(
  target: AssistantMessage["usage"],
  value: unknown,
): void {
  const usage = value as AnthropicUsageFields;
  if (usage.input_tokens != null) target.input = usage.input_tokens;
  if (usage.output_tokens != null) target.output = usage.output_tokens;
  if (usage.cache_read_input_tokens != null) {
    target.cacheRead = usage.cache_read_input_tokens;
  }
  if (usage.cache_creation_input_tokens != null) {
    target.cacheWrite = usage.cache_creation_input_tokens;
  }
  const cacheWrite1h = readCacheWrite1h(value);
  if (cacheWrite1h !== undefined) target.cacheWrite1h = cacheWrite1h;
  const thinkingTokens = usage.output_tokens_details?.thinking_tokens;
  if (thinkingTokens != null) target.reasoning = thinkingTokens;
  target.totalTokens =
    target.input + target.output + target.cacheRead + target.cacheWrite;
}

export function getAllowedFallbackModels(
  model: Model<Api>,
): AnthropicAllowedFallbackModel[] {
  return (
    model.compat as
      | { allowedFallbackModels?: AnthropicAllowedFallbackModel[] }
      | undefined
  )?.allowedFallbackModels ?? [];
}

export function resolveUsageModel(
  model: Model<Api>,
  responseModel: string,
): Model<Api> {
  if (responseModel === model.id) return model;
  const fallback = getAllowedFallbackModels(model).find(
    (candidate) =>
      candidate.provider === model.provider && candidate.model === responseModel,
  );
  return fallback
    ? { ...model, id: responseModel, cost: fallback.cost }
    : model;
}

function betaValues(value: string | undefined): string[] {
  return (value ?? "")
    .split(",")
    .map((beta) => beta.trim())
    .filter((beta) => beta && !BLOCKED_BETAS.has(beta));
}

export function makeDefaultHeaders(
  isOAuth: boolean,
  options?: SimpleStreamOptions,
  serverSideFallback = false,
): Record<string, string> {
  const headers: Record<string, string> = {
    accept: "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
  };
  const betas = new Set<string>(
    isOAuth ? REQUIRED_BETAS : ["interleaved-thinking-2025-05-14"],
  );

  if (isOAuth) {
    if (process.env.PI_CACHE_RETENTION === "long") {
      betas.add("extended-cache-ttl-2025-04-11");
    }
    headers["user-agent"] = USER_AGENT;
    headers["x-app"] = "cli";
  }
  if (serverSideFallback) betas.add(SERVER_SIDE_FALLBACK_BETA);

  if (options?.headers) {
    for (const [key, value] of Object.entries(options.headers)) {
      const normalizedKey = key.toLowerCase();
      if (
        isOAuth &&
        (normalizedKey === "x-api-key" || normalizedKey === "authorization")
      ) {
        continue;
      }
      if (normalizedKey === "anthropic-beta") {
        if (typeof value === "string") {
          for (const beta of betaValues(value)) betas.add(beta);
        }
        continue;
      }
      const existingKey = Object.keys(headers).find(
        (header) => header.toLowerCase() === normalizedKey,
      );
      if (existingKey) delete headers[existingKey];
      if (value !== null) headers[key] = value;
    }
  }

  headers["anthropic-beta"] = [...betas]
    .filter((beta) => !BLOCKED_BETAS.has(beta))
    .join(",");
  return headers;
}

export async function prepareAnthropicRequest(
  model: Model<Api>,
  context: Context,
  options: SimpleStreamOptions | undefined,
  isOAuth: boolean,
): Promise<AnthropicRequestParams> {
  const maxTokens = options?.maxTokens || Math.floor(model.maxTokens / 3);
  let params: AnthropicRequestParams = {
    model: model.id,
    messages: convertPiMessagesToAnthropic(context.messages, isOAuth, model),
    max_tokens: maxTokens,
    stream: true,
  };

  const system = buildAnthropicSystemPrompt(context.systemPrompt, isOAuth);
  if (system) params.system = system as never;
  if (context.tools?.length) {
    params.tools = convertPiToolsToAnthropic(context.tools, isOAuth);
  }

  if (options?.reasoning && model.reasoning && maxTokens > 1) {
    const defaultBudgets: Record<string, number> = {
      minimal: 1024,
      low: 4096,
      medium: 10240,
      high: 20480,
      xhigh: 32000,
    };
    const customBudget =
      options.thinkingBudgets?.[
        options.reasoning as keyof typeof options.thinkingBudgets
      ];
    const requestedBudget =
      customBudget ?? defaultBudgets[options.reasoning] ?? 10240;

    params.thinking = {
      type: "enabled",
      budget_tokens: Math.min(requestedBudget, maxTokens - 1),
    };
  }

  if (options?.toolChoice) {
    params.tool_choice = { type: options.toolChoice };
  }
  const fallbacks = getAllowedFallbackModels(model);
  if (fallbacks.length > 0) {
    params.fallbacks = fallbacks.map((fallback) => ({ model: fallback.model }));
  }

  const replacement = await options?.onPayload?.(params, model);
  if (replacement !== undefined) {
    params = replacement as AnthropicRequestParams;
  }
  return params;
}

export function streamAnthropicOAuth(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  // A new real request supersedes any pending keepalive ping. Its generation
  // also prevents an older concurrent completion from publishing stale params.
  const cacheRequestGeneration = cacheKeepalive.beginRequest();
  const stream = createAssistantMessageEventStream();

  void (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    try {
      const apiKey = options?.apiKey;
      if (!apiKey) {
        throw new Error(
          "No Anthropic auth available. Run /login and choose Claude Pro/Max.",
        );
      }

      const isOAuth = isClaudeOAuthAccessToken(apiKey);
      const allowedFallbackModels = getAllowedFallbackModels(model);
      const defaultHeaders = makeDefaultHeaders(
        isOAuth,
        options,
        allowedFallbackModels.length > 0,
      );

      if (isOAuth) defaultHeaders.authorization = `Bearer ${apiKey}`;

      const client = new Anthropic({
        baseURL: model.baseUrl,
        apiKey: isOAuth ? null : apiKey,
        authToken: isOAuth ? apiKey : null,
        defaultHeaders,
        dangerouslyAllowBrowser: true,
        fetch: options?.fetch,
      });

      const params = await prepareAnthropicRequest(
        model,
        context,
        options,
        isOAuth,
      );
      let usageModel = model;

      // Raw stream instead of the MessageStream helper: MessageStream
      // accumulates tool_use input and JSON.parses it on content_block_stop,
      // which throws under fine-grained-tool-streaming (input may be invalid
      // mid-flight) and aborts the turn. The raw stream yields the same
      // RawMessageStreamEvents; tool args are already parsed leniently below.
      const providerRequestStartedAt = Date.now();
      const { data: anthropicStream, response: httpResponse } =
        await client.messages
          .create(params, {
            signal: options?.signal,
          })
          .withResponse();

      if (options?.onResponse) {
        try {
          await options.onResponse(
            {
              status: httpResponse.status,
              headers: headersToRecord(httpResponse.headers),
            },
            model,
          );
        } catch {
          // Response hooks are best-effort and should not break streaming.
        }
      }

      stream.push({ type: "start", partial: output });

      const blocks = output.content as IndexedBlock[];

      for await (const event of anthropicStream) {
        if (event.type === "message_start") {
          output.responseId = event.message.id;
          output.model = event.message.model;
          usageModel = resolveUsageModel(model, output.model);
          applyAnthropicUsage(output.usage, event.message.usage);
          calculateCost(usageModel, output.usage);
          continue;
        }

        if (event.type === "content_block_start") {
          if (event.content_block.type === "text") {
            output.content.push({
              type: "text",
              text: "",
              index: event.index,
            } as IndexedBlock);
            stream.push({
              type: "text_start",
              contentIndex: output.content.length - 1,
              partial: output,
            });
          } else if (event.content_block.type === "thinking") {
            output.content.push({
              type: "thinking",
              thinking: "",
              thinkingSignature: "",
              index: event.index,
            } as IndexedBlock);
            stream.push({
              type: "thinking_start",
              contentIndex: output.content.length - 1,
              partial: output,
            });
          } else if (event.content_block.type === "redacted_thinking") {
            output.content.push({
              type: "thinking",
              thinking: "[Reasoning redacted]",
              thinkingSignature: event.content_block.data,
              redacted: true,
              index: event.index,
            } as IndexedBlock);
            stream.push({
              type: "thinking_start",
              contentIndex: output.content.length - 1,
              partial: output,
            });
          } else if (event.content_block.type === "tool_use") {
            output.content.push({
              type: "toolCall",
              id: event.content_block.id,
              name: isOAuth
                ? fromClaudeCodeToolName(
                    event.content_block.name,
                    context.tools,
                  )
                : event.content_block.name,
              arguments: {},
              partialJson: "",
              index: event.index,
            } as IndexedBlock);
            stream.push({
              type: "toolcall_start",
              contentIndex: output.content.length - 1,
              partial: output,
            });
          }
          continue;
        }

        if (event.type === "content_block_delta") {
          const contentIndex = blocks.findIndex(
            (block) => block.index === event.index,
          );
          const block = blocks[contentIndex];
          if (!block) continue;

          if (event.delta.type === "text_delta" && block.type === "text") {
            block.text += event.delta.text;
            stream.push({
              type: "text_delta",
              contentIndex,
              delta: event.delta.text,
              partial: output,
            });
          } else if (
            event.delta.type === "thinking_delta" &&
            block.type === "thinking"
          ) {
            block.thinking += event.delta.thinking;
            stream.push({
              type: "thinking_delta",
              contentIndex,
              delta: event.delta.thinking,
              partial: output,
            });
          } else if (
            event.delta.type === "signature_delta" &&
            block.type === "thinking"
          ) {
            block.thinkingSignature =
              (block.thinkingSignature || "") + event.delta.signature;
          } else if (
            event.delta.type === "input_json_delta" &&
            block.type === "toolCall"
          ) {
            block.partialJson += event.delta.partial_json;
            try {
              block.arguments = JSON.parse(block.partialJson) as Record<
                string,
                unknown
              >;
            } catch {}
            stream.push({
              type: "toolcall_delta",
              contentIndex,
              delta: event.delta.partial_json,
              partial: output,
            });
          }
          continue;
        }

        if (event.type === "content_block_stop") {
          const contentIndex = blocks.findIndex(
            (block) => block.index === event.index,
          );
          const block = blocks[contentIndex];
          if (!block) continue;

          delete (block as { index?: number }).index;
          if (block.type === "text") {
            stream.push({
              type: "text_end",
              contentIndex,
              content: block.text,
              partial: output,
            });
          } else if (block.type === "thinking") {
            stream.push({
              type: "thinking_end",
              contentIndex,
              content: block.thinking,
              partial: output,
            });
          } else if (block.type === "toolCall") {
            try {
              block.arguments = JSON.parse(block.partialJson) as Record<
                string,
                unknown
              >;
            } catch {}
            delete (block as { partialJson?: string }).partialJson;
            stream.push({
              type: "toolcall_end",
              contentIndex,
              toolCall: block,
              partial: output,
            });
          }
          continue;
        }

        if (event.type === "message_delta") {
          output.stopReason = mapStopReason(event.delta.stop_reason);
          applyAnthropicUsage(output.usage, event.usage);
          calculateCost(usageModel, output.usage);
        }
      }

      if (options?.signal?.aborted) throw new Error("Request aborted");
      stream.push({
        type: "done",
        reason: output.stopReason as "stop" | "length" | "toolUse",
        message: output,
      });
      stream.end();
      cacheKeepalive.finishRequest(
        cacheRequestGeneration,
        { client, params },
        output.usage.input + output.usage.cacheRead + output.usage.cacheWrite,
        providerRequestStartedAt,
        resolveKeepalivePolicy(),
      );
    } catch (error) {
      for (const block of output.content as Array<{
        index?: number;
        partialJson?: string;
      }>) {
        delete block.index;
        delete block.partialJson;
      }
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage =
        error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}
