import assert from "node:assert/strict";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readlinkSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { calculateCost } from "@earendil-works/pi-ai";
import { convertPiMessagesToAnthropic } from "../.test-dist/src/convert.js";
import {
  ensureClaudeCodeAgentAlias,
  registerAnthropicProvider,
  registerCacheShutdown,
} from "../.test-dist/src/index.js";
import { buildAnthropicSystemPrompt } from "../.test-dist/src/prompt.js";
import {
  applyAnthropicUsage,
  keepaliveDebug,
  makeDefaultHeaders,
  pingAnthropicCache,
  prepareAnthropicRequest,
  readCacheWrite1h,
  resolveUsageModel,
  streamAnthropicOAuth,
} from "../.test-dist/src/stream.js";

function withLongRetention(run) {
  const previous = process.env.PI_CACHE_RETENTION;
  process.env.PI_CACHE_RETENTION = "long";
  try {
    return run();
  } finally {
    if (previous === undefined) delete process.env.PI_CACHE_RETENTION;
    else process.env.PI_CACHE_RETENTION = previous;
  }
}

function model(overrides = {}) {
  return {
    id: "claude-fable-5",
    name: "Claude Fable 5",
    api: "anthropic-messages",
    provider: "anthropic",
    baseUrl: "https://api.anthropic.com",
    reasoning: false,
    input: ["text"],
    cost: { input: 1, output: 5, cacheRead: 0.1, cacheWrite: 1.25 },
    contextWindow: 200_000,
    maxTokens: 8_192,
    ...overrides,
  };
}

function sse(events) {
  return events
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    .join("");
}

test("long retention marks system and message cache breakpoints", () => {
  withLongRetention(() => {
    const system = buildAnthropicSystemPrompt("Stable instructions", true);
    assert.ok(system);
    assert.deepEqual(system.at(-1).cache_control, { type: "ephemeral", ttl: "1h" });

    const messages = convertPiMessagesToAnthropic(
      [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 0 }],
      true,
      { provider: "anthropic", api: "anthropic-messages", id: "claude-haiku-4-5" },
    );
    const finalContent = messages.at(-1).content;
    assert.ok(Array.isArray(finalContent));
    assert.deepEqual(finalContent.at(-1).cache_control, { type: "ephemeral", ttl: "1h" });
  });
});

test("long retention adds the Anthropic extended-cache beta", () => {
  withLongRetention(() => {
    const headers = makeDefaultHeaders(true);
    assert.match(headers["anthropic-beta"], /extended-cache-ttl-2025-04-11/);
  });
});

test("header merge preserves required betas and rejects fine-grained streaming", () => {
  withLongRetention(() => {
    const headers = makeDefaultHeaders(
      true,
      {
        headers: {
          "Anthropic-Beta": [
            "fast-mode-2026-02-01",
            "fine-grained-tool-streaming-2025-05-14",
          ].join(","),
          Authorization: "Bearer must-not-win",
          "X-API-Key": "must-not-win",
        },
      },
      true,
    );
    const betas = new Set(headers["anthropic-beta"].split(","));
    for (const required of [
      "claude-code-20250219",
      "oauth-2025-04-20",
      "interleaved-thinking-2025-05-14",
      "extended-cache-ttl-2025-04-11",
      "server-side-fallback-2026-07-01",
      "fast-mode-2026-02-01",
    ]) {
      assert.equal(betas.has(required), true, `missing ${required}`);
    }
    assert.equal(betas.has("fine-grained-tool-streaming-2025-05-14"), false);
    assert.equal(Object.keys(headers).some((key) => key.toLowerCase() === "authorization"), false);
    assert.equal(Object.keys(headers).some((key) => key.toLowerCase() === "x-api-key"), false);
  });
});

test("request preparation honors tool choice, fallbacks, and payload replacement", async () => {
  const requested = model({
    compat: {
      allowedFallbackModels: [
        {
          provider: "anthropic",
          model: "claude-opus-4-8",
          cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
        },
      ],
    },
  });
  let hookPayload;
  const params = await prepareAnthropicRequest(
    requested,
    {
      systemPrompt: "Stable instructions",
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 0 }],
      tools: [
        {
          name: "read",
          description: "Read a file",
          parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
        },
      ],
    },
    {
      toolChoice: "none",
      onPayload(payload) {
        hookPayload = payload;
        return { ...payload, speed: "fast" };
      },
    },
    true,
  );

  assert.deepEqual(hookPayload.tool_choice, { type: "none" });
  assert.deepEqual(hookPayload.fallbacks, [{ model: "claude-opus-4-8" }]);
  assert.equal(params.speed, "fast");
  assert.deepEqual(params.tool_choice, { type: "none" });
  assert.equal(params.tools[0].name, "Read");
});

test("payload hook may mutate in place and return undefined", async () => {
  let hookPayload;
  const params = await prepareAnthropicRequest(
    model(),
    {
      systemPrompt: "",
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 0 }],
    },
    {
      toolChoice: "auto",
      onPayload(payload) {
        hookPayload = payload;
        payload.speed = "fast";
      },
    },
    true,
  );
  assert.strictEqual(params, hookPayload);
  assert.equal(params.speed, "fast");
  assert.deepEqual(params.tool_choice, { type: "auto" });
  assert.equal("fallbacks" in params, false);

  const plain = await prepareAnthropicRequest(
    model(),
    { systemPrompt: "", messages: [] },
    undefined,
    true,
  );
  assert.equal("tool_choice" in plain, false);
  assert.equal("fallbacks" in plain, false);
});

test("returned fallback model supplies response identity and pricing", () => {
  const requested = model({
    compat: {
      allowedFallbackModels: [
        {
          provider: "anthropic",
          model: "claude-opus-4-8",
          cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
        },
      ],
    },
  });
  const fallback = resolveUsageModel(requested, "claude-opus-4-8");
  assert.equal(fallback.id, "claude-opus-4-8");
  assert.deepEqual(fallback.cost, { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 });

  const usage = {
    input: 1_000_000,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 1_000_000,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  calculateCost(fallback, usage);
  assert.equal(usage.cost.input, 3);
  assert.equal(resolveUsageModel(requested, "unknown-model"), requested);

  const crossProvider = model({
    compat: {
      allowedFallbackModels: [
        {
          provider: "other-provider",
          model: "same-id",
          cost: { input: 99, output: 99, cacheRead: 99, cacheWrite: 99 },
        },
      ],
    },
  });
  assert.equal(resolveUsageModel(crossProvider, "same-id"), crossProvider);
});

test("usage deltas preserve fields omitted after message_start", () => {
  const usage = {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  applyAnthropicUsage(usage, {
    input_tokens: 100,
    output_tokens: 1,
    cache_read_input_tokens: 80,
    cache_creation_input_tokens: 20,
    cache_creation: { ephemeral_1h_input_tokens: 20 },
  });
  applyAnthropicUsage(usage, {
    input_tokens: null,
    output_tokens: 12,
    cache_read_input_tokens: null,
  });
  assert.deepEqual(
    {
      input: usage.input,
      output: usage.output,
      cacheRead: usage.cacheRead,
      cacheWrite: usage.cacheWrite,
      cacheWrite1h: usage.cacheWrite1h,
      totalTokens: usage.totalTokens,
    },
    {
      input: 100,
      output: 12,
      cacheRead: 80,
      cacheWrite: 20,
      cacheWrite1h: 20,
      totalTokens: 212,
    },
  );
});

test("stream sends transformed fallbacks and keeps fallback-priced cache usage", async () => {
  let requestBody;
  let requestHeaders;
  const requested = model({
    compat: {
      allowedFallbackModels: [
        {
          provider: "anthropic",
          model: "claude-opus-4-8",
          cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
        },
      ],
    },
  });
  const stream = streamAnthropicOAuth(
    requested,
    {
      systemPrompt: "Stable instructions",
      messages: [{ role: "user", content: [{ type: "text", text: "hello" }], timestamp: 0 }],
    },
    {
      apiKey: "sk-ant-oat-test",
      toolChoice: "none",
      headers: {
        "anthropic-beta": "fast-mode-2026-02-01,fine-grained-tool-streaming-2025-05-14",
      },
      onPayload(payload) {
        return { ...payload, speed: "fast" };
      },
      async fetch(input, init) {
        const request = input instanceof Request ? input : new Request(input, init);
        requestHeaders = new Headers(request.headers);
        requestBody = JSON.parse(await request.clone().text());
        return new Response(
          sse([
            {
              type: "message_start",
              message: {
                id: "msg_fallback",
                type: "message",
                role: "assistant",
                model: "claude-opus-4-8",
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: {
                  input_tokens: 100,
                  output_tokens: 1,
                  cache_read_input_tokens: 80,
                  cache_creation_input_tokens: 20,
                  cache_creation: { ephemeral_1h_input_tokens: 20 },
                },
              },
            },
            {
              type: "content_block_start",
              index: 0,
              content_block: { type: "text", text: "" },
            },
            {
              type: "content_block_delta",
              index: 0,
              delta: { type: "text_delta", text: "ok" },
            },
            { type: "content_block_stop", index: 0 },
            {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: 2 },
            },
            { type: "message_stop" },
          ]),
          { status: 200, headers: { "content-type": "text/event-stream" } },
        );
      },
    },
  );

  const result = await stream.result();
  assert.equal(requestBody.speed, "fast");
  assert.deepEqual(requestBody.tool_choice, { type: "none" });
  assert.deepEqual(requestBody.fallbacks, [{ model: "claude-opus-4-8" }]);
  const betas = new Set(requestHeaders.get("anthropic-beta").split(","));
  assert.equal(betas.has("server-side-fallback-2026-07-01"), true);
  assert.equal(betas.has("fast-mode-2026-02-01"), true);
  assert.equal(betas.has("fine-grained-tool-streaming-2025-05-14"), false);
  assert.equal(result.responseId, "msg_fallback");
  assert.equal(result.model, "claude-opus-4-8");
  assert.equal(result.content[0].text, "ok");
  assert.deepEqual(
    {
      input: result.usage.input,
      output: result.usage.output,
      cacheRead: result.usage.cacheRead,
      cacheWrite: result.usage.cacheWrite,
      cacheWrite1h: result.usage.cacheWrite1h,
    },
    { input: 100, output: 2, cacheRead: 80, cacheWrite: 20, cacheWrite1h: 20 },
  );
  assert.ok(Math.abs(result.usage.cost.input - 0.0003) < 1e-12);
});

test("cacheWrite1h parser accepts only finite numeric usage", () => {
  assert.equal(
    readCacheWrite1h({ cache_creation: { ephemeral_1h_input_tokens: 12_345 } }),
    12_345,
  );
  assert.equal(readCacheWrite1h({ cache_creation: { ephemeral_1h_input_tokens: "123" } }), undefined);
  assert.equal(readCacheWrite1h({ cache_creation: { ephemeral_1h_input_tokens: Infinity } }), undefined);
  assert.equal(readCacheWrite1h(undefined), undefined);
});

test("Anthropic ping replays the exact params and forwards its abort signal", async () => {
  const params = { model: "test", messages: [], max_tokens: 1, stream: true };
  const controller = new AbortController();
  let createCalls = 0;
  const client = {
    messages: {
      create(receivedParams, options) {
        createCalls += 1;
        assert.strictEqual(receivedParams, params);
        assert.strictEqual(options.signal, controller.signal);
        return (async function* events() {
          yield {
            type: "message_start",
            message: {
              usage: {
                cache_read_input_tokens: 12_345,
                cache_creation_input_tokens: 0,
              },
            },
          };
          throw new Error("ping should stop after message_start");
        })();
      },
    },
  };

  const result = await pingAnthropicCache({ client, params }, controller.signal);
  assert.deepEqual(result, { cacheRead: 12_345, cacheWrite: 0 });
  assert.equal(createCalls, 1);
});

test("debug logging is opt-in, private, bounded-path, and nofollow", () => {
  const configDir = mkdtempSync(join(tmpdir(), "pi-oauth-debug-"));
  const previousDir = process.env.PI_CODING_AGENT_DIR;
  const previousDebug = process.env.PI_CACHE_KEEPALIVE_DEBUG;
  const logFile = join(configDir, "cache", "cache-keepalive.log");
  try {
    process.env.PI_CODING_AGENT_DIR = configDir;
    process.env.PI_CACHE_KEEPALIVE_DEBUG = "0";
    keepaliveDebug("must not be written");
    assert.equal(existsSync(logFile), false);

    process.env.PI_CACHE_KEEPALIVE_DEBUG = "1";
    keepaliveDebug("line one\nline two");
    assert.equal(statSync(logFile).mode & 0o777, 0o600);
    assert.match(readFileSync(logFile, "utf8"), /line one line two\n$/);

    rmSync(logFile);
    const target = join(configDir, "symlink-target");
    writeFileSync(target, "unchanged", { mode: 0o600 });
    symlinkSync(target, logFile);
    keepaliveDebug("must not follow");
    assert.equal(readFileSync(target, "utf8"), "unchanged");
  } finally {
    if (previousDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previousDir;
    if (previousDebug === undefined) delete process.env.PI_CACHE_KEEPALIVE_DEBUG;
    else process.env.PI_CACHE_KEEPALIVE_DEBUG = previousDebug;
    rmSync(configDir, { recursive: true, force: true });
  }
});

test("compatibility alias exposes only the Pi agent directory", () => {
  const home = mkdtempSync(join(tmpdir(), "pi-oauth-plus-"));
  try {
    const target = join(home, ".pi", "agent");
    mkdirSync(target, { recursive: true });
    ensureClaudeCodeAgentAlias(home);

    const aliasRoot = join(home, ".Claude Code");
    const alias = join(aliasRoot, "agent");
    assert.equal(lstatSync(aliasRoot).isDirectory(), true);
    assert.equal(lstatSync(aliasRoot).mode & 0o077, 0);
    assert.equal(lstatSync(alias).isSymbolicLink(), true);
    assert.equal(readlinkSync(alias), target);
  } finally {
    rmSync(home, { recursive: true, force: true });
  }
});

test("provider registration declares subscription auth and shutdown cleanup", () => {
  let providerConfig;
  let shutdown;
  const pi = {
    registerProvider(name, config) {
      assert.equal(name, "anthropic");
      providerConfig = config;
    },
    on(event, callback) {
      assert.equal(event, "session_shutdown");
      shutdown = callback;
    },
  };

  registerAnthropicProvider(pi);
  assert.equal(providerConfig.oauth.isSubscription, true);
  assert.equal(typeof providerConfig.streamSimple, "function");
  assert.equal(typeof shutdown, "function");
});

test("session shutdown registration invokes keepalive cancellation", () => {
  let handler;
  let cancellations = 0;
  const pi = {
    on(event, callback) {
      assert.equal(event, "session_shutdown");
      handler = callback;
    },
  };

  registerCacheShutdown(pi, () => {
    cancellations += 1;
  });
  assert.equal(typeof handler, "function");
  handler();
  assert.equal(cancellations, 1);
});

test("one-hour writes use two-times input pricing", () => {
  const usage = {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 100_000,
    cacheWrite1h: 100_000,
    totalTokens: 100_000,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  const model = {
    id: "test",
    name: "Test",
    api: "anthropic-messages",
    provider: "anthropic",
    baseUrl: "https://api.anthropic.com",
    reasoning: false,
    input: ["text"],
    cost: { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
    contextWindow: 200_000,
    maxTokens: 8_192,
  };

  calculateCost(model, usage);
  assert.equal(usage.cost.cacheWrite, 2);
  assert.equal(usage.cost.total, 2);
});
