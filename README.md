# pi-anthropic-oauth-plus

[pi-anthropic-oauth](https://www.npmjs.com/package/pi-anthropic-oauth) (Claude
Pro/Max OAuth for [pi](https://github.com/badlogic/pi-mono), MIT, by leohenon)
plus prompt-cache upgrades. The first commit is the pristine upstream 0.2.4;
everything on top is this repo's work — see the diff.

## What's added

### Pi 0.84.3 provider compatibility

- honors Pi's `onPayload` replacement hook, so request-body extensions such as
  `speed: "fast"` reach Anthropic;
- maps provider-neutral `toolChoice` and declares OAuth as subscription-backed;
- sends configured server-side fallback models and prices usage against the
  model Anthropic actually returns;
- merges caller beta headers with the required OAuth/interleaved/long-cache
  betas instead of replacing them;
- always removes fine-grained tool streaming because its raw partial JSON can
  corrupt large edit arguments.

### 1-hour cache TTL (`PI_CACHE_RETENTION=long`)

Upstream hardcodes `cache_control: { type: "ephemeral" }` (5-minute TTL), so
returning to a session after >5 minutes re-bills the whole prompt. With
`PI_CACHE_RETENTION=long` this fork sends `ttl: "1h"` on every cache
breakpoint plus the `extended-cache-ttl-2025-04-11` beta. Measured on real
usage: ~76% of cache misses eliminated (most idle gaps are 5–30 min).

### Cache keepalive (replay-and-abort)

Even 1h dies over lunch. After each successful turn the extension targets a
ping 55 minutes after the provider request **started**: it re-sends the exact
same request object and stops after `message_start`. A confirmed cache read
refreshes the TTL for another hour. Guards:

- exact replay only (same client, tools, system, messages, and thinking config);
- request-start TTL accounting, matching Anthropic's documented semantics;
- capped at `PI_CACHE_KEEPALIVE` pings (default 6, up to ~6.5h total coverage;
  `0` disables);
- only a positive `cache_read_input_tokens` result rearms the next ping;
- expiry, provider errors, zero cache reads, real requests, reload, session
  switch, and shutdown all stop and abort pending work;
- stale request/ping completions cannot replace or cancel newer state;
- prompts under 10k tokens are skipped;
- out-of-band: nothing is emitted to pi, so ping usage is also absent from Pi's
  session/footer cost totals;
- timers are `unref()`ed so oneshot `pi -p` processes exit normally.

Extra knobs: `PI_CACHE_KEEPALIVE_DELAY_MS`, `PI_CACHE_KEEPALIVE_DEBUG=1`.
Debug output is capped at 64 KiB in the private
`~/.pi/agent/cache/cache-keepalive.log` file (or the configured Pi directory).
The delay override must remain below the 1-hour TTL, and the ping override is
capped at 24; invalid values fall back to the safe defaults.

### `cacheWrite1h` usage accounting

Parses `cache_creation.ephemeral_1h_input_tokens` into `usage.cacheWrite1h`,
so you can verify from the session file that 1h writes actually happen.

## Install

```bash
pi remove npm:pi-anthropic-oauth        # if the upstream package is installed
pi install git:github.com/duy-tung/pi-anthropic-oauth-plus@v0.3.2
export PI_CACHE_RETENTION=long          # e.g. in ~/.zshenv
```

Version 0.3.2 requires Pi 0.84.3 or newer.

### Prompt-rewrite compatibility alias

The upstream provider's aggressive identity rewrite can transform
`~/.pi/agent` text in the system prompt into `~/.Claude Code/agent`. On a new
installation this fork creates a private `~/.Claude Code` directory containing
only `agent -> ~/.pi/agent`; it does not alias the entire `~/.pi` state tree.
An existing upstream-style `~/.Claude Code -> ~/.pi` symlink is left untouched
rather than rewritten or deleted silently.

## Cost model and operating boundary

- 1h cache writes cost 2× base input (vs 1.25× for 5m); reads stay 0.1×.
- Each keepalive ping costs one cache read of the prompt. It is generally worth
  extending a large live conversation through lunch or an afternoon break, but
  not indefinitely warming an abandoned session.
- Six pings cover about 390 minutes from the real request start. Sleep, Pi
  restart, OAuth/network failure, provider eviction, or a longer gap can still
  produce a legitimate miss.
- For overnight or >6h breaks, use Pi's `/compact`, a handoff, or a new session
  instead of increasing keepalive toward 24 hours.
- Pi's `Cache miss after … idle` label compares visible assistant-request
  timestamps and does not know about these out-of-band pings.

## Development

```bash
npm install
npm run check
npm test
npm run pack:check
```

Tests use a fake clock and fake provider; they do not wait for real TTLs or make
model requests.

## License

MIT — upstream copyright leohenon, modifications copyright duy-tung.
