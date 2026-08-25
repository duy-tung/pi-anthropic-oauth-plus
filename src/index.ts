import { existsSync, lstatSync, mkdirSync, symlinkSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { OAuthCredentials } from "@earendil-works/pi-ai";
import { loginAnthropic, refreshAnthropicToken } from "./auth.js";
import { cancelCacheKeepalive, streamAnthropicOAuth } from "./stream.js";

export function ensureClaudeCodeAgentAlias(home: string = homedir()): void {
  // Aggressive OAuth prompt rewriting can turn `~/.pi/agent` into
  // `~/.Claude Code/agent`. Preserve that path without aliasing the whole
  // `~/.pi` state tree on new installations.
  const target = join(home, ".pi", "agent");
  const aliasRoot = join(home, ".Claude Code");
  const alias = join(aliasRoot, "agent");
  if (!existsSync(target) || existsSync(alias)) return;
  try {
    if (!existsSync(aliasRoot)) mkdirSync(aliasRoot, { mode: 0o700 });
    if (!lstatSync(aliasRoot).isDirectory()) return;
    symlinkSync(target, alias);
  } catch {
    // Compatibility setup is best-effort and must not block provider loading.
  }
}

export function registerCacheShutdown(
  pi: ExtensionAPI,
  cancel: () => void = cancelCacheKeepalive,
): void {
  pi.on("session_shutdown", () => cancel());
}

export function registerAnthropicProvider(pi: ExtensionAPI): void {
  pi.registerProvider("anthropic", {
    baseUrl: "https://api.anthropic.com",
    api: "anthropic-messages",
    oauth: {
      name: "Claude Pro/Max",
      isSubscription: true,
      usesCallbackServer: true,
      login: loginAnthropic,
      refreshToken: refreshAnthropicToken,
      getApiKey: (credentials: OAuthCredentials) => credentials.access,
    },
    streamSimple: streamAnthropicOAuth,
  });

  registerCacheShutdown(pi);
}

export default function (pi: ExtensionAPI) {
  ensureClaudeCodeAgentAlias();
  registerAnthropicProvider(pi);
}
