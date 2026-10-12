"""Self-test for context-audit v0.1.0 — run: python3 selftest.py

No gateway, no Hermes runtime, no third-party deps: the plugin module is loaded
straight from __init__.py and pointed at an in-memory SQLite DB. One case uses a
temp *file* DB to prove that tool arguments and results never reach disk.

Exits non-zero on the first failing assertion set, so it doubles as a CI gate.
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("context_audit", os.path.join(HERE, "__init__.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PASS = 0
FAILS = []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"PASS: {name}")
    else:
        FAILS.append(name)
        print(f"FAIL: {name} {detail}")


def fresh_db():
    """Point the plugin at a clean in-memory DB and clear per-turn scratch."""
    mod.DB_PATH = ":memory:"
    mod._DB = mod.AuditDB(":memory:")
    mod._turn_state.clear()
    return mod._DB


def q(sql, args=()):
    return mod._DB._conn.execute(sql, args).fetchall()


# ===========================================================================
# 1. logging
# ===========================================================================
print("\n-- logging --")
fresh_db()

# -- pre_tool_call sizes the args and flags the oversized ones
big = {"command": "x" * (mod.CONTEXT_EATER_BYTES + 10)}
small = {"command": "echo hi"}
mod.on_pre_tool_call(tool_name="terminal", args=big, session_id="s1", turn_id="t1",
                     tool_call_id="call_big")
mod.on_pre_tool_call(tool_name="read_file", args=small, session_id="s1", turn_id="t1",
                     tool_call_id="call_small")
rows = q("SELECT tool_name, args_bytes, oversized, resolved FROM tool_calls ORDER BY id")
check("pre_tool_call records both calls", len(rows) == 2, rows)
check("oversized arg flagged", rows[0][2] == 1 and rows[1][2] == 0, rows)
check("args_bytes is the serialized size", rows[0][1] == mod._payload_bytes(big), rows[0][1])
check("rows start unresolved", rows[0][3] == 0 and rows[1][3] == 0, rows)

# -- post_tool_call attaches the result size by exact tool_call_id
mod.on_post_tool_call(tool_name="read_file", result="y" * 4096, session_id="s1", turn_id="t1",
                      tool_call_id="call_small", duration_ms=7, status="ok")
r = q("SELECT result_bytes, duration_ms, status, resolved FROM tool_calls WHERE tool_call_id='call_small'")
check("post confirms by exact id", r == [(4096, 7, "ok", 1)], r)

# -- FIFO pairing: pre with an empty id, post with a real one (agent-ledger's v0.1 bug)
mod.on_pre_tool_call(tool_name="write_file", args={"path": "/a"}, session_id="s2", tool_call_id="")
time.sleep(0.001)
mod.on_pre_tool_call(tool_name="write_file", args={"path": "/b"}, session_id="s2", tool_call_id="")
mod.on_post_tool_call(tool_name="write_file", result="ok", session_id="s2", tool_call_id="call_X")
first, second = q("SELECT args_bytes, resolved FROM tool_calls WHERE tool_name='write_file' ORDER BY id")
check("FIFO confirms the OLDER row first", first[1] == 1 and second[1] == 0, (first, second))
mod.on_post_tool_call(tool_name="write_file", result="ok", session_id="s2", tool_call_id="call_Y")
first, second = q("SELECT resolved FROM tool_calls WHERE tool_name='write_file' ORDER BY id")
check("second post takes the next row", first[0] == 1 and second[0] == 1, (first, second))

# -- FIFO fallback prefers the same turn when turn_id is supplied
mod.on_pre_tool_call(tool_name="terminal", args={"cmd": "t1"}, session_id="s4", turn_id="turnA", tool_call_id="")
time.sleep(0.001)
mod.on_pre_tool_call(tool_name="terminal", args={"cmd": "t2"}, session_id="s4", turn_id="turnB", tool_call_id="")
mod.on_post_tool_call(tool_name="terminal", result="ok", session_id="s4", turn_id="turnB", tool_call_id="call_Z")
row = q("SELECT turn_id, resolved FROM tool_calls WHERE tool_name='terminal' AND session_id='s4' ORDER BY id")
check("turn-scoped fallback confirms the matching turn's row",
      row[0][1] == 0 and row[1][1] == 1 and row[1][0] == "turnB", row)

# -- performance indexes exist
idx = [r[0] for r in q("SELECT name FROM sqlite_master WHERE type='index'")]
check("session/turn index on tool_calls", "idx_tool_calls_session_turn" in idx, idx)
check("session index on compactions", "idx_compactions_session" in idx, idx)

# -- post without any pre is still recorded (never crashes, never strands)
before = q("SELECT COUNT(*) FROM tool_calls")[0][0]
mod.on_post_tool_call(tool_name="cronjob", result="done", session_id="s3", tool_call_id="call_orphan")
after = q("SELECT COUNT(*) FROM tool_calls")[0][0]
check("post-without-pre recorded", after == before + 1, (before, after))

# -- non-string results must not blow up
mod.on_post_tool_call(tool_name="execute_code", result={"output": "z" * 100}, session_id="s3")
r = q("SELECT result_bytes FROM tool_calls WHERE tool_name='execute_code'")
check("dict result sized, not stored", r and r[0][0] > 100, r)

# ===========================================================================
# 2. per-turn row + compaction
# ===========================================================================
print("\n-- turn rollup --")
fresh_db()
mod.on_pre_llm_call(session_id="s9", turn_id="t9")
time.sleep(0.001)
mod.on_pre_tool_call(tool_name="web_extract", args={"url": "http://x"}, session_id="s9", turn_id="t9")
mod.on_post_tool_call(tool_name="web_extract", result="w" * 60000, session_id="s9", turn_id="t9")
mod.on_pre_tool_call(tool_name="read_file", args={"path": "/p"}, session_id="s9", turn_id="t9")
mod.on_post_tool_call(tool_name="read_file", result="r" * 500, session_id="s9", turn_id="t9")
mod.on_post_auxiliary_call(aux_task="title_generation", session_id="s9")
mod.on_post_auxiliary_call(aux_task="compression", session_id="s9", model="deepseek")
history = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 400}]
mod.on_post_llm_call(session_id="s9", turn_id="t9", conversation_history=history,
                     assistant_response="a" * 400, model="deepseek-ai/DeepSeek-V4.1-Flash",
                     platform="telegram")

turn = q("SELECT est_input_tokens, est_output_tokens, largest_tool, largest_tool_bytes,"
         " compaction_events, model, platform FROM turns")
check("one turn row written", len(turn) == 1, turn)
check("input token estimate > 0", turn and turn[0][0] > 0, turn)
check("output token estimate > 0", turn and turn[0][1] > 0, turn)
check("heaviest tool = web_extract", turn and turn[0][2] == "web_extract", turn)
check("heaviest tool bytes = its result", turn and turn[0][3] == 60000, turn)
check("compaction counted, title_generation ignored", turn and turn[0][4] == 1, turn)
check("model/platform persisted", turn and turn[0][5].startswith("deepseek") and turn[0][6] == "telegram", turn)

# -- a torn-down turn (no post_tool_call ever fired) still reports its args weight
fresh_db()
mod.on_pre_llm_call(session_id="s10", turn_id="t10")
args10 = {"command": "y" * 5000}
mod.on_pre_tool_call(tool_name="terminal", args=args10, session_id="s10", turn_id="t10")
mod.on_post_llm_call(session_id="s10", turn_id="t10", conversation_history=[], assistant_response="")
t10 = q("SELECT largest_tool, largest_tool_bytes FROM turns")
check("torn-down turn falls back to args size", t10 == [("terminal", mod._payload_bytes(args10))], t10)

# -- verify the registered hook names against the real Hermes allow-list
class _Ctx:
    def __init__(self):
        self.hooks, self.commands = [], []
    def register_hook(self, name, fn):
        self.hooks.append(name)
    def register_command(self, name, fn, description="", args_hint=""):
        self.commands.append(name)


ctx = _Ctx()
mod.register(ctx)
check("register() wires 5 hooks", len(ctx.hooks) == 5, ctx.hooks)
check("register() wires /context-audit", ctx.commands == ["context-audit"], ctx.commands)
try:
    sys.path.insert(0, "/root/.hermes/hermes-agent")
    from hermes_cli.plugins import VALID_HOOKS  # type: ignore
except Exception:
    print("SKIP: hermes_cli not importable here — hook names checked against plugin.yaml only")
else:
    unknown = [h for h in ctx.hooks if h not in VALID_HOOKS]
    check("every hook exists in Hermes VALID_HOOKS", not unknown, unknown)

# ===========================================================================
# 3. report aggregation
# ===========================================================================
print("\n-- report aggregation --")
db = fresh_db()
now = time.time()
db.add_tool_call("web_extract", 100, session_id="r1", ts=now - 3600)
db.confirm_tool_call("web_extract", session_id="r1", result_bytes=900, status="ok")
db.add_tool_call("web_extract", 100, session_id="r1", ts=now - 3500)
db.confirm_tool_call("web_extract", session_id="r1", result_bytes=1100, status="ok")
db.add_tool_call("read_file", 50, session_id="r1", ts=now - 3400)
db.confirm_tool_call("read_file", session_id="r1", result_bytes=150, status="ok")
db.add_tool_call("terminal", mod.CONTEXT_EATER_BYTES + 1, session_id="r1", ts=now - 3300)
db.add_tool_call("ancient", 999999, session_id="r1", ts=now - 30 * 86400)

eaters = db.top_eaters(now - 7 * 86400, limit=5)
totals = [e["total_bytes"] for e in eaters]
check("ranking is by total bytes desc", totals == sorted(totals, reverse=True),
      [(e["tool_name"], e["total_bytes"]) for e in eaters])
check("bigger args outrank smaller result-only tool",
      [e["tool_name"] for e in eaters][:2] == ["terminal", "web_extract"],
      [e["tool_name"] for e in eaters])
we = next(e for e in eaters if e["tool_name"] == "web_extract")
check("totals summed over calls", we["total_bytes"] == (100 + 900) + (100 + 1100), we)
check("call count correct", we["calls"] == 2, we)
check("avg per call correct", we["avg_bytes"] == we["total_bytes"] // 2, we)
check("oversized call counted per tool", next(e for e in eaters if e["tool_name"] == "terminal")["oversized_calls"] == 1,
      eaters)
check("out-of-window rows excluded", "ancient" not in [e["tool_name"] for e in eaters], eaters)
check("limit respected", len(db.top_eaters(now - 7 * 86400, limit=2)) == 2)

summary = db.summary(now - 7 * 86400)
check("summary counts turns/calls/oversized/compactions",
      summary["turns"] == 0 and summary["tool_calls"] == 4 and summary["oversized_calls"] == 1
      and summary["compactions"] == 0, summary)

text = mod.render_report(days=7)
check("report names the top eater", "web_extract" in text, text[:120])
check("report carries one mitigation per item", text.count("->") == len(mod.report(days=7)["eaters"]), text)
check("mitigation is tool-specific", "char_limit" in mod.mitigation_for("web_extract", 1024),
      mod.mitigation_for("web_extract", 1024))
check("unknown tool still gets a mitigation", "heaviest context source" in mod.mitigation_for("mystery", 10))

empty = mod.render_report(days=7, db=mod.AuditDB(":memory:"))
check("empty window says so", "No tool calls" in empty, empty)

check("_parse_args default", mod._parse_args("") == (7, 5, False))
check("_parse_args days", mod._parse_args("30") == (30, 5, False))
check("_parse_args limit + json", mod._parse_args("14 limit=3 json") == (14, 3, True))
check("_parse_args tolerates junk", mod._parse_args("banana 5") == (5, 5, False))
check("_parse_args clamps", mod._parse_args("9999 limit=999") == (365, 50, False))
check("json command output parses", json.loads(mod.context_audit_command("json"))["plugin"] == "context-audit")

# ===========================================================================
# 4. metadata only — nothing but sizes reaches disk
# ===========================================================================
print("\n-- privacy --")
tmpdir = tempfile.mkdtemp()
file_db = os.path.join(tmpdir, "audit.db")
mod.DB_PATH = file_db
mod._DB = mod.AuditDB(file_db)
ARGS_SENTINEL = "SENTINEL_ARGS_SECRET_9f3a"
RESULT_SENTINEL = "SENTINEL_RESULT_SECRET_7b1c"
mod.on_pre_tool_call(tool_name="terminal", args={"command": ARGS_SENTINEL},
                     session_id="p1", turn_id="p1")
mod.on_post_tool_call(tool_name="terminal", result=RESULT_SENTINEL * 50, session_id="p1", turn_id="p1")
mod._DB._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
mod._DB.close()
blob = open(file_db, "rb").read()
check("args content absent from DB file", ARGS_SENTINEL.encode() not in blob)
check("result content absent from DB file", RESULT_SENTINEL.encode() not in blob)
check("sizes still present", mod.AuditDB(file_db).summary(0)["tool_calls"] == 1)

# ===========================================================================
# 5. fail-open
# ===========================================================================
print("\n-- fail-open --")
mod.DB_PATH = "/proc/definitely-not-writable/audit.db"
mod._DB = None
mod._warned = False
raised = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        mod.on_pre_tool_call(tool_name="terminal", args={"command": "x"}, session_id="f1")
        mod.on_pre_tool_call(tool_name="terminal", args={"command": "x"}, session_id="f1")
        out = mod.context_audit_command("")
    except Exception as exc:  # pragma: no cover - the assertion below is the point
        raised = exc
runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
check("hooks never raise on a broken DB", raised is None, raised)
check("RuntimeWarning emitted exactly once", len(runtime_warnings) == 1, len(runtime_warnings))
check("command returns a string instead of raising", isinstance(out, str) and "unavailable" in out, out)
check("_warned latched", mod._warned is True)

# ===========================================================================
print(f"\n{PASS} passed, {len(FAILS)} failed")
if FAILS:
    for name in FAILS:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
