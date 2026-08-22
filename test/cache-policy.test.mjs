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
  registerCacheShutdown,
} from "../.test-dist/src/index.js";
import { buildAnthropicSystemPrompt } from "../.test-dist/src/prompt.js";
import {
  keepaliveDebug,
  makeDefaultHeaders,
  pingAnthropicCache,
  readCacheWrite1h,
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
