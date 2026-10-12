# context-audit

Context-window forensics for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Agent sessions get fat and nobody can say why. Tool outputs are enormous, compaction
fires out of nowhere, a `web_extract` quietly returns 40 KB that then rides along in
every subsequent request. Existing plugins mostly track **cost**; this one tracks
**what filled the window**.

`context-audit` records metadata only — tool name, byte length, ids, timestamps — and
answers one question: *which tools ate my context, and what do I do about it?*

Observer-only. Python stdlib only. Fails open.

## Install

```bash
# from the catalog (once published)
hermes plugins install context-audit

# or straight from a clone
git clone https://github.com/ZidoCode/context-audit ~/.hermes/plugins/context-audit
hermes plugins enable context-audit
```

Verify the contracts against your Hermes version before trusting it with anything:

```bash
hermes plugins validate ~/.hermes/plugins/context-audit   # catalog admission gate
hermes plugins doctor   ~/.hermes/plugins/context-audit   # real runtime contracts
python3 ~/.hermes/plugins/context-audit/selftest.py       # 47 checks, in-memory SQLite, no gateway
python3 ~/.hermes/plugins/context-audit/test_integration.py  # real discovery + dispatch, throwaway profile
```

`test_integration.py` copies the plugin into a temp `HERMES_HOME`, lets Hermes
discover it, fires the hooks through `invoke_hook` and reads back the DB — proving
discovery, dispatch and path resolution, which the unit suite cannot. It skips
(exit 0) when the Hermes source tree is not importable.

`hermes plugins enable context-audit` activates the hooks in a **running** gateway
immediately — it only reports `restart_required` when no gateway answered. No restart
needed. Hooks see turns that *start* after activation: a turn already in flight gets no
`pre_llm_call`, so its `turns` row appears only when that turn finishes.

## Use

```
/context-audit            # top 5 eaters, last 7 days
/context-audit 30         # last 30 days
/context-audit 30 limit=3 # narrower ranking
/context-audit json       # machine-readable
```

Real output from a seeded demo DB (`render_report(days=7)`):

```
CONTEXT AUDIT — top 5 context eaters, last 7 day(s)
turns=0  tool_calls=48  oversized(>50.0 KB args)=1  compactions=3

1. web_extract           416.6 KB      9 calls  avg 46.3 KB
   -> avg 46.3 KB per call — lower char_limit / prefer js()+page_info() over full-page dumps
2. terminal              333.6 KB     15 calls  avg 22.2 KB  [1 oversized]
   -> avg 22.2 KB per call — redirect bulk output to a file and read a slice instead
3. read_file              67.3 KB     11 calls  avg 6.1 KB
   -> avg 6.1 KB per call — pass offset+limit; never paste whole files into context
4. execute_code           44.0 KB      4 calls  avg 11.0 KB
   -> avg 11.0 KB per call — redirect bulk output to a file and read a slice instead
5. browser_exec           16.2 KB      3 calls  avg 5.4 KB
   -> avg 5.4 KB per call — lower char_limit / prefer js()+page_info() over full-page dumps

3 compaction event(s) in this window — a rising count means the window is filling faster than it drains.
```

## What it records

| table | one row per | columns |
|---|---|---|
| `turns` | LLM turn | `est_input_tokens`, `est_output_tokens`, `largest_tool`, `largest_tool_bytes`, `compaction_events`, `model`, `platform` |
| `tool_calls` | tool call | `tool_name`, `args_bytes`, `result_bytes`, `duration_ms`, `status`, `oversized`, `session_id`, `turn_id`, `tool_call_id` |
| `compactions` | compaction | `session_id`, `aux_task`, `model` |

DB: `<HERMES_HOME>/plugin-data/context-audit/audit.db` (WAL). Follows the active
profile through `get_hermes_home()`.

Hooks: `pre_tool_call` (sizes args, flags >50 KB), `post_tool_call` (sizes the result),
`pre_llm_call` (marks turn start), `post_llm_call` (writes the turn row),
`post_auxiliary_call` (compaction events).

## Privacy

Only byte **lengths** are computed and stored. Arguments and results are serialized
in memory inside the hook and immediately discarded — `selftest.py` asserts both by
inserting sentinel strings and grepping the resulting DB file for them.

## Honest limitations

1. **Token counts are estimates** — chars ÷ 4. Good for ranking turns against each
   other, not for billing or for predicting an exact context overflow.
2. **Compaction detection is indirect.** Hermes 0.21.x has no dedicated compaction
   hook, so a compaction is counted when `post_auxiliary_call` reports
   `aux_task == "compression"`. A compaction that aborts mid-call records nothing,
   and any deterministic fallback compaction Hermes performs *without* a summariser
   LLM call is invisible to this plugin.
3. **"Largest tool" is output bytes, not tokens.** It ranks by recorded result
   size, and falls back to the largest *args* size only for a turn that recorded no
   result at all (torn down before any `post_tool_call`) — so a killed turn still
   shows its weight, but as a different quantity.
4. **Per-call pairing is best-effort.** Exact `tool_call_id` match first, then the
   oldest unresolved row for the same `(tool_name, session_id)` — FIFO, never LIFO.
   Hermes can fire pre/post with an empty id on one side only. Aggregate totals stay
   correct; individual call attribution can swap if the same tool is called
   concurrently in one session.
5. **Attribution is per tool name only.** No call-site, no argument breakdown. MCP
   tools show up under their full `mcp__server__tool` name.
6. **The DB is never pruned.** Metadata only (~100 B/row), but it grows forever. A
   10 000-call history is roughly 1 MB. Delete the file to reset.
7. **No session filter and no UI.** `/context-audit` reports a wall-clock window
   across all sessions; `json` output is the machine-readable escape hatch.
8. **The estimate excludes the system prompt**, which is fixed per session and
   measured from `conversation_history` only. If your system prompt is the eater,
   this tool will not tell you that.

## Uninstall

```bash
hermes plugins disable context-audit    # config-only: a RUNNING gateway keeps the hooks wired until restart
rm -rf ~/.hermes/plugins/context-audit ~/.hermes/plugin-data/context-audit
```

Disabling writes config; there is no un-wire primitive, so an already-running gateway
keeps calling these hooks (and writing to the DB) until it restarts. To stop the
auditing *now*, delete `~/.hermes/plugin-data/context-audit/` after the gateway
restarts — not before.

## License

MIT — see [LICENSE](LICENSE).
