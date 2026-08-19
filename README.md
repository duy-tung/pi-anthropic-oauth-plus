# pi-anthropic-oauth-plus

[pi-anthropic-oauth](https://www.npmjs.com/package/pi-anthropic-oauth) (Claude
Pro/Max OAuth for [pi](https://github.com/badlogic/pi-mono), MIT, by leohenon)
plus prompt-cache upgrades. The first commit is the pristine upstream 0.2.4;
everything on top is this repo's work — see the diff.

## What's added

### 1-hour cache TTL (`PI_CACHE_RETENTION=long`)

Upstream hardcodes `cache_control: { type: "ephemeral" }` (5-minute TTL), so
returning to a session after >5 minutes re-bills the whole prompt. With
`PI_CACHE_RETENTION=long` this fork sends `ttl: "1h"` on every cache
breakpoint plus the `extended-cache-ttl-2025-04-11` beta. Measured on real
usage: ~76% of cache misses eliminated (most idle gaps are 5–30 min).

### Cache keepalive (replay-and-abort)

Even 1h dies over lunch. After each successful turn the extension arms a timer
(default 55 min): it re-sends the byte-identical last request and aborts right
after `message_start` — the prompt is a pure cache *read* (0.1×), which
refreshes the TTL for another hour. Guards:

- byte-identical replay only (same tools/system/messages/thinking config);
- never fires if the cache already expired (would be a pointless 2× re-write);
- capped at `PI_CACHE_KEEPALIVE` pings (default 3 ≈ keeps cache warm ~4h; `0` disables);
- prompts under 10k tokens are not worth pinging and are skipped;
- out-of-band: nothing is emitted to pi, the session file is untouched;
- `unref()`ed timer, so `pi -p` oneshot runs exit normally.

Extra knobs: `PI_CACHE_KEEPALIVE_DELAY_MS`, `PI_CACHE_KEEPALIVE_DEBUG=1`
(logs to `/tmp/pi-cache-keepalive.log`).

### `cacheWrite1h` usage accounting

Parses `cache_creation.ephemeral_1h_input_tokens` into `usage.cacheWrite1h`,
so you can verify from the session file that 1h writes actually happen.

## Install

```bash
pi remove npm:pi-anthropic-oauth        # if the upstream package is installed
pi install git:github.com/duy-tung/pi-anthropic-oauth-plus
export PI_CACHE_RETENTION=long          # e.g. in ~/.zshenv
```

## Cost model

- 1h cache writes cost 2× base input (vs 1.25× for 5m); reads stay 0.1×.
- Worth it when you regularly return to a session after 5–60 min idle.
- Each keepalive ping costs one cache read of the prompt (0.1×) — profitable
  whenever the chance you return within the hour exceeds ~5–10%.

## License

MIT — upstream copyright leohenon, modifications copyright duy-tung.
