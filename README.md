# context-audit

Context-window forensics for Hermes Agent — find out where your context window actually goes.

## What it does

- **Per-turn token estimates** (input/output) via `post_llm_call`
- **Largest tool output per turn** — name + byte count, never contents
- **Oversized tool call detection** — args over 50 KB flagged as context-eater candidates
- **Compaction event log**
- `/context-audit` command — top 5 context eaters with one mitigation each

## Install

```bash
hermes plugins install context-audit   # once listed in the catalog
```

Or clone into `~/.hermes/plugins/context-audit/` and run `hermes plugins enable context-audit`.

## Design guarantees

- **Observer-only** — never blocks or modifies a tool call; fails open
- **Metadata only** — tool name, byte length, ids, timestamps. Never args or results.
- **stdlib only** — sqlite3, json, no dependencies
- **Profile-aware** — data lives under `HERMES_HOME/plugin-data/context-audit/`

## Verify

```bash
python3 selftest.py        # 47 checks
python3 test_integration.py  # real discovery+dispatch through Hermes loader
hermes plugins validate .
```

Live-verified: hooks fire on a running gateway, DB populated with real session/turn ids.

MIT license. Maintained by [ZidoCode](https://github.com/ZidoCode).
