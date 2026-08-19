import { appendFileSync } from "node:fs";
import Anthropic from "@anthropic-ai/sdk";
import type { MessageCreateParamsStreaming } from "@anthropic-ai/sdk/resources/messages.js";
import {
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

const REQUIRED_BETAS = [
  "claude-code-20250219",
  "oauth-2025-04-20",
  // fine-grained-tool-streaming removed: it ships the model's raw, unvalidated
  // tool-input JSON. For large edits full of quotes/newlines the streamed
  // string escaping breaks, so a field (e.g. edit.oldText) swallows the rest of
  // the structure — surfacing as either a hard JSON.parse crash or a wrong-shape
  // schema-validation failure. Default streaming has the server validate/buffer
  // tool JSON, guaranteeing well-formed, correctly-structured input.
  "interleaved-thinking-2025-05-14",
] as const;

// ---------------------------------------------------------------------------
// Cache keepalive: with PI_CACHE_RETENTION=long (1h TTL), an idle session's
// prompt cache dies after 1h. Re-sending the byte-identical last request and
// aborting right after `message_start` refreshes the TTL for the price of a
// cache read (0.1x) instead of a full 2x re-write on the next real request.
//
// Guards:
// - only replays the exact params of the last successful request (same tools,
//   system, messages, thinking config) so the cache lookup hits;
// - never runs if the cache is already expired (sleep/wake, timer drift) —
//   that would be a pointless 2x re-write;
// - capped at PI_CACHE_KEEPALIVE pings (default 3, 0 disables);
// - out-of-band: nothing is emitted to pi, the session file is untouched;
// - timer is unref()ed so oneshot `pi -p` processes exit normally.
// ---------------------------------------------------------------------------

const KEEPALIVE_CACHE_TTL_MS = 60 * 60 * 1000;
const KEEPALIVE_MIN_PROMPT_TOKENS = 10_000;

type KeepaliveState = {
  timer?: NodeJS.Timeout;
  count: number;
  maxPings: number;
  lastAccessAt: number;
  client: Anthropic;
  params: MessageCreateParamsStreaming;
};

let keepalive: KeepaliveState | undefined;

function keepaliveMaxPings(): number {
  const raw = process.env.PI_CACHE_KEEPALIVE;
  if (raw === undefined || raw.trim() === "") return 3;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 3;
}

function keepaliveDelayMs(): number {
  const raw = process.env.PI_CACHE_KEEPALIVE_DELAY_MS;
  const n = raw ? Number(raw) : NaN;
  return Number.isFinite(n) && n > 0 ? n : 55 * 60 * 1000;
}

function keepaliveDebug(message: string): void {
  if (!process.env.PI_CACHE_KEEPALIVE_DEBUG) return;
  try {
    appendFileSync(
      "/tmp/pi-cache-keepalive.log",
      `${new Date().toISOString()} ${message}\n`,
    );
  } catch {}
}

function cancelKeepalive(): void {
  if (keepalive?.timer) clearTimeout(keepalive.timer);
  keepalive = undefined;
}

function scheduleKeepalive(
  client: Anthropic,
  params: MessageCreateParamsStreaming,
  promptTokens: number,
): void {
  cancelKeepalive();
  if (process.env.PI_CACHE_RETENTION !== "long") return;
  const maxPings = keepaliveMaxPings();
  if (maxPings === 0) return;
  // Tiny prompts are below the model's min cacheable length or simply not
  // worth refreshing.
  if (promptTokens < KEEPALIVE_MIN_PROMPT_TOKENS) return;
  keepalive = {
    count: 0,
    maxPings,
    lastAccessAt: Date.now(),
    client,
    params,
  };
  armKeepaliveTimer();
  keepaliveDebug(`scheduled promptTokens=${promptTokens} maxPings=${maxPings}`);
}

function armKeepaliveTimer(): void {
  if (!keepalive) return;
  const timer = setTimeout(() => {
    void runKeepalivePing();
  }, keepaliveDelayMs());
  timer.unref?.();
  keepalive.timer = timer;
}

async function runKeepalivePing(): Promise<void> {
  const state = keepalive;
  if (!state) return;
  // Cache already expired: a ping now would be a full 2x cache re-write for a
  // session nobody may come back to. Give up instead.
  if (Date.now() - state.lastAccessAt >= KEEPALIVE_CACHE_TTL_MS) {
    keepaliveDebug("skip: cache TTL already expired");
    cancelKeepalive();
    return;
  }
  try {
    const stream = await state.client.messages.create(state.params, {});
    for await (const event of stream) {
      if (event.type === "message_start") {
        const usage = (event as { message?: { usage?: Record<string, unknown> } })
          .message?.usage as
          | { cache_read_input_tokens?: number; cache_creation_input_tokens?: number }
          | undefined;
        keepaliveDebug(
          `ping ok cacheRead=${usage?.cache_read_input_tokens ?? 0} cacheWrite=${usage?.cache_creation_input_tokens ?? 0}`,
        );
        break; // Prompt processed, cache refreshed; stop paying for output.
      }
    }
  } catch (error) {
    // Expired OAuth token, network trouble, API rejection: stop quietly.
    keepaliveDebug(`ping failed: ${error instanceof Error ? error.message : String(error)}`);
    cancelKeepalive();
    return;
  }
  // A real request may have superseded this state while the ping was in flight.
  if (keepalive !== state) return;
  state.count += 1;
  state.lastAccessAt = Date.now();
  if (state.count >= state.maxPings) {
    keepaliveDebug(`done: reached maxPings=${state.maxPings}`);
    cancelKeepalive();
    return;
  }
  armKeepaliveTimer();
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

function makeDefaultHeaders(
  isOAuth: boolean,
  options?: SimpleStreamOptions,
): Record<string, string> {
  const headers: Record<string, string> = {
    accept: "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
  };

  if (isOAuth) {
    const betas = [...REQUIRED_BETAS];
    if (process.env.PI_CACHE_RETENTION === "long") {
      betas.push("extended-cache-ttl-2025-04-11");
    }
    headers["anthropic-beta"] = betas.join(",");
    headers["user-agent"] = USER_AGENT;
    headers["x-app"] = "cli";
  } else {
    headers["anthropic-beta"] = ["interleaved-thinking-2025-05-14"].join(",");
  }

  if (options?.headers) {
    for (const [key, value] of Object.entries(options.headers)) {
      const normalizedKey = key.toLowerCase();
      if (
        isOAuth &&
        (normalizedKey === "x-api-key" || normalizedKey === "authorization")
      ) {
        continue;
      }
      const existingKey = Object.keys(headers).find(
        (header) => header.toLowerCase() === normalizedKey,
      );
      if (existingKey) delete headers[existingKey];
      if (value !== null) headers[key] = value;
    }
  }

  return headers;
}

export function streamAnthropicOAuth(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  // A new real request supersedes any pending keepalive ping.
  cancelKeepalive();
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
      const defaultHeaders = makeDefaultHeaders(isOAuth, options);

      if (isOAuth) defaultHeaders.authorization = `Bearer ${apiKey}`;

      const client = new Anthropic({
        baseURL: model.baseUrl,
        apiKey: isOAuth ? null : apiKey,
        authToken: isOAuth ? apiKey : null,
        defaultHeaders,
        dangerouslyAllowBrowser: true,
      });

      const maxTokens =
        options?.maxTokens || Math.floor(model.maxTokens / 3);

      const params: MessageCreateParamsStreaming = {
        model: model.id,
        messages: convertPiMessagesToAnthropic(context.messages, isOAuth, model),
        max_tokens: maxTokens,
        stream: true,
      };

      const system = buildAnthropicSystemPrompt(context.systemPrompt, isOAuth);
      if (system) params.system = system as never;
      if (context.tools?.length)
        params.tools = convertPiToolsToAnthropic(context.tools, isOAuth);

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

      // Raw stream instead of the MessageStream helper: MessageStream
      // accumulates tool_use input and JSON.parses it on content_block_stop,
      // which throws under fine-grained-tool-streaming (input may be invalid
      // mid-flight) and aborts the turn. The raw stream yields the same
      // RawMessageStreamEvents; tool args are already parsed leniently below.
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
          output.usage.input = event.message.usage.input_tokens || 0;
          output.usage.output = event.message.usage.output_tokens || 0;
          output.usage.cacheRead =
            (event.message.usage as { cache_read_input_tokens?: number })
              .cache_read_input_tokens || 0;
          output.usage.cacheWrite =
            (event.message.usage as { cache_creation_input_tokens?: number })
              .cache_creation_input_tokens || 0;
          (output.usage as any).cacheWrite1h =
            (event.message.usage as any).cache_creation
              ?.ephemeral_1h_input_tokens || 0;
          output.usage.totalTokens =
            output.usage.input +
            output.usage.output +
            output.usage.cacheRead +
            output.usage.cacheWrite;
          calculateCost(model, output.usage);
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
          output.usage.input =
            (event.usage as { input_tokens?: number }).input_tokens ||
            output.usage.input;
          output.usage.output =
            (event.usage as { output_tokens?: number }).output_tokens ||
            output.usage.output;
          output.usage.cacheRead =
            (event.usage as { cache_read_input_tokens?: number })
              .cache_read_input_tokens || 0;
          output.usage.cacheWrite =
            (event.usage as { cache_creation_input_tokens?: number })
              .cache_creation_input_tokens || 0;
          const cc1h = (event.usage as any).cache_creation
            ?.ephemeral_1h_input_tokens;
          if (cc1h != null) (output.usage as any).cacheWrite1h = cc1h;
          const thinkingTokens = (
            event.usage as {
              output_tokens_details?: { thinking_tokens?: number };
            }
          ).output_tokens_details?.thinking_tokens;
          if (thinkingTokens != null) {
            output.usage.reasoning = thinkingTokens;
          }
          output.usage.totalTokens =
            output.usage.input +
            output.usage.output +
            output.usage.cacheRead +
            output.usage.cacheWrite;
          calculateCost(model, output.usage);
        }
      }

      if (options?.signal?.aborted) throw new Error("Request aborted");
      stream.push({
        type: "done",
        reason: output.stopReason as "stop" | "length" | "toolUse",
        message: output,
      });
      stream.end();
      scheduleKeepalive(
        client,
        params,
        output.usage.input + output.usage.cacheRead + output.usage.cacheWrite,
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
