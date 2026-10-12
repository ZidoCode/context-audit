"""Integration test for context-audit — run: python3 test_integration.py

Proves what selftest.py cannot: Hermes really discovers this plugin, `invoke_hook`
really reaches these callbacks, and the DB path really resolves through
get_hermes_home(). Runs against a throwaway HERMES_HOME under the OS temp dir, so
the live profile is never touched.

Needs the Hermes source tree importable (`hermes_cli`). Exits 0 with SKIP when it
is absent, so it can never false-fail on a machine without Hermes installed.
"""
import os
import shutil
import sqlite3
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_NAME = "context-audit"
HOOKS = ["pre_tool_call", "post_tool_call", "pre_llm_call", "post_llm_call", "post_auxiliary_call"]

FAILS = []


def check(name, cond, detail=""):
    print(("PASS: " if cond else "FAIL: ") + name + (f" {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def find_repo():
    """Locate the hermes-agent source tree, or None when it isn't installed."""
    home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
    for cand in (os.environ.get("HERMES_AGENT_REPO"),
                 os.path.join(home, "hermes-agent"),
                 "/home/hatch/.hermes/hermes-agent"):
        if cand and os.path.isfile(os.path.join(cand, "hermes_cli", "plugins_loader.py")):
            return cand
    return None


def main() -> int:
    repo = find_repo()
    if repo is None:
        print("SKIP: hermes-agent source tree not found (set HERMES_AGENT_REPO)")
        return 0
    sys.path.insert(0, repo)

    scratch = tempfile.mkdtemp(prefix="context-audit-it-")
    # HERMES_HOME must be set BEFORE hermes_cli resolves it at import time.
    os.environ["HERMES_HOME"] = scratch
    try:
        os.makedirs(os.path.join(scratch, "plugins"))
        shutil.copytree(HERE, os.path.join(scratch, "plugins", PLUGIN_NAME),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        with open(os.path.join(scratch, "config.yaml"), "w") as fh:
            fh.write("_config_version: 49\nplugins:\n  enabled:\n    - %s\n  disabled: []\n" % PLUGIN_NAME)

        from hermes_cli import plugins as P
        from hermes_cli.lifecycle import invoke_hook

        P.discover_plugins(force=True)
        loaded = (getattr(P.get_plugin_manager(), "_plugins", {}) or {})
        check("plugin discovered in a fresh profile", PLUGIN_NAME in loaded, sorted(loaded))
        entry = loaded.get(PLUGIN_NAME)
        check("plugin loaded without error", entry is not None and not getattr(entry, "error", None),
              getattr(entry, "error", None))

        # Drive the real dispatch path — the same calls the agent makes at runtime.
        invoke_hook("pre_llm_call", session_id="it", turn_id="turn1", conversation_history=[],
                    user_message="hi", is_first_turn=True, model="m", platform="cli")
        invoke_hook("pre_tool_call", tool_name="web_extract", args={"url": "x", "pad": "p" * 3000},
                    session_id="it", tool_call_id="c1", turn_id="turn1")
        invoke_hook("post_tool_call", tool_name="web_extract", args={"url": "x"}, result="R" * 9000,
                    session_id="it", tool_call_id="c1", turn_id="turn1", duration_ms=12, status="ok")
        invoke_hook("pre_tool_call", tool_name="terminal", args={"command": "z" * (51 * 1024)},
                    session_id="it", tool_call_id="c2", turn_id="turn1")
        invoke_hook("post_auxiliary_call", aux_task="compression", session_id="it", model="m")
        invoke_hook("post_llm_call", session_id="it", turn_id="turn1",
                    conversation_history=[{"role": "user", "content": "u" * 800}],
                    assistant_response="a" * 400, model="m", platform="cli")

        db_path = os.path.join(scratch, "plugin-data", PLUGIN_NAME, "audit.db")
        check("audit.db created under the profile's plugin-data", os.path.exists(db_path), db_path)
        if not os.path.exists(db_path):
            return 1

        con = sqlite3.connect(db_path)
        calls = con.execute("SELECT tool_name, result_bytes, oversized FROM tool_calls ORDER BY id").fetchall()
        turns = con.execute("SELECT largest_tool, largest_tool_bytes, compaction_events FROM turns").fetchall()
        comps = con.execute("SELECT COUNT(*) FROM compactions").fetchone()[0]

        check("two tool calls recorded", len(calls) == 2, calls)
        check("result size attached by the real post hook", calls and calls[0][1] == 9000, calls)
        check("oversized args flagged", any(c[2] == 1 for c in calls), calls)
        check("turn row attributes the largest OUTPUT", turns and turns[0][0] == "web_extract"
              and turns[0][1] == 9000, turns)
        check("compaction counted on the turn", turns and turns[0][2] == 1, turns)
        check("compaction row written", comps == 1, comps)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + str(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
