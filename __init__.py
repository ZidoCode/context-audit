"""context-audit — context-window forensics for Hermes Agent.

Tracks where an agent session's context window actually goes: per-turn token
estimates, the single largest tool output of each turn, oversized tool calls,
and compaction events. Everything lands in a local SQLite DB as metadata only.

Design rules (non-negotiable for this plugin):

* **Observer-only.** No hook ever returns a directive. Tool calls are never
  blocked, modified, reordered or delayed.
* **Fail open.** Any exception inside a hook or the report is swallowed and
  reported once as a ``RuntimeWarning``; the agent path is never broken by the
  audit.
* **Python stdlib only** — ``sqlite3``, ``json``, ``re``, ``warnings``,
  ``threading``, ``functools``, ``os``, ``time``, ``datetime``.
* **Metadata only.** Byte lengths are computed in memory and only the *length*
  is stored. Tool arguments and tool results are never written to disk.

Data lives in ``<HERMES_HOME>/plugin-data/context-audit/audit.db``.

Read the report in-session with ``/context-audit`` (``/context-audit 30`` for a
30-day window, ``/context-audit json`` for machine-readable output), or call
:func:`render_report` / :func:`report` directly.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

__version__ = "0.1.0"

#: A tool call whose serialized args exceed this many bytes is flagged as a
#: "context eater" candidate (args are replayed into history on later turns).
CONTEXT_EATER_BYTES = 50 * 1024

PLUGIN_NAME = "context-audit"
DB_FILENAME = "audit.db"

#: Rough token estimate. Deliberately crude and documented as such: the point is
#: relative attribution between turns, not a billing-grade tokenizer.
CHARS_PER_TOKEN = 4

_WINDOW_DAYS = 7
_REPORT_LIMIT = 5

_warned = False
_state_lock = threading.Lock()
# Per-turn scratch: (session_id, turn_id) -> {"started_ts", "hooked"}. Never
# persisted, never holds content.
_turn_state: dict[tuple[str, str], dict] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 REAL    NOT NULL,
    session_id         TEXT    NOT NULL DEFAULT '',
    turn_id            TEXT    NOT NULL DEFAULT '',
    model              TEXT    NOT NULL DEFAULT '',
    platform           TEXT    NOT NULL DEFAULT '',
    est_input_tokens   INTEGER NOT NULL DEFAULT 0,
    est_output_tokens  INTEGER NOT NULL DEFAULT 0,
    largest_tool       TEXT    NOT NULL DEFAULT '',
    largest_tool_bytes INTEGER NOT NULL DEFAULT 0,
    compaction_events  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);

CREATE TABLE IF NOT EXISTS tool_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    tool_call_id TEXT    NOT NULL DEFAULT '',
    session_id   TEXT    NOT NULL DEFAULT '',
    turn_id      TEXT    NOT NULL DEFAULT '',
    tool_name    TEXT    NOT NULL DEFAULT '',
    args_bytes   INTEGER NOT NULL DEFAULT 0,
    result_bytes INTEGER NOT NULL DEFAULT 0,
    duration_ms  INTEGER,
    status       TEXT    NOT NULL DEFAULT '',
    oversized    INTEGER NOT NULL DEFAULT 0,
    resolved     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts   ON tool_calls(ts);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session_turn ON tool_calls(session_id, turn_id);

CREATE TABLE IF NOT EXISTS compactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    aux_task   TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_compactions_ts ON compactions(ts);
CREATE INDEX IF NOT EXISTS idx_compactions_session ON compactions(session_id, ts);
"""


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _fail_open(exc: BaseException) -> None:
    """Swallow an observer error, warning exactly once per process."""
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        f"{PLUGIN_NAME}: observer error, auditing degraded ({type(exc).__name__}: {exc})",
        RuntimeWarning,
        stacklevel=3,
    )


def _observer(fn):
    """Wrap a hook so it can never raise into the agent path."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - observers must never propagate
            _fail_open(exc)
        return None
    return wrapper


def _hermes_home() -> Path:
    """Resolve HERMES_HOME through the runtime helper, degrading to the env var."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        env = os.environ.get("HERMES_HOME")
        return Path(env) if env else Path.home() / ".hermes"


def _default_db_path() -> str:
    return str(_hermes_home() / "plugin-data" / PLUGIN_NAME / DB_FILENAME)


#: Overridable so tests (and the selftest) can point at a temp file or ``:memory:``.
DB_PATH = _default_db_path()


def _payload_bytes(obj) -> int:
    """Serialized size of a tool-args dict, in bytes. Contents are discarded."""
    if obj is None:
        return 0
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8", "replace"))
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8", "replace"))
    except Exception:
        return len(str(obj).encode("utf-8", "replace"))


_MESSAGE_CONTENT_ATTRS = ("content", "text", "reasoning", "reasoning_content")


def _message_chars(message) -> int:
    """Character count of one history message, without keeping its content."""
    if message is None:
        return 0
    if isinstance(message, str):
        return len(message)
    if isinstance(message, dict):
        total = 0
        for key, value in message.items():
            if isinstance(value, str):
                total += len(value)
            elif isinstance(value, (list, dict)):
                total += len(json.dumps(value, ensure_ascii=False, default=str))
        return total
    total = 0
    for attr in _MESSAGE_CONTENT_ATTRS:
        value = getattr(message, attr, None)
        if isinstance(value, str):
            total += len(value)
    if not total:
        return len(str(message))
    return total


def _est_tokens(value) -> int:
    """Crude char/4 token estimate over a message, list of messages, or string."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        chars = sum(_message_chars(m) for m in value)
    else:
        chars = _message_chars(value)
    if not chars:
        return 0
    return max(1, chars // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

class AuditDB:
    """SQLite-backed metadata store. One row per turn / tool call / compaction."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._memory = path == ":memory:"
        if not self._memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        if not self._memory:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- writes ----------------------------------------------------------

    def add_tool_call(self, tool_name: str, args_bytes: int, session_id: str = "",
                      turn_id: str = "", tool_call_id: str = "", ts: float | None = None) -> int:
        """Record a tool call *before* it runs. Returns the row id."""
        args_bytes = int(args_bytes or 0)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO tool_calls (ts, tool_call_id, session_id, turn_id, tool_name,"
                " args_bytes, oversized) VALUES (?,?,?,?,?,?,?)",
                (ts if ts is not None else _now(), tool_call_id or "", session_id or "",
                 turn_id or "", tool_name or "", args_bytes,
                 1 if args_bytes > CONTEXT_EATER_BYTES else 0),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def confirm_tool_call(self, tool_name: str, session_id: str = "", tool_call_id: str = "",
                          result_bytes: int = 0, duration_ms=None, status: str = "",
                          turn_id: str = "") -> int:
        """Attach the result size to the matching pre_tool_call row.

        Exact ``tool_call_id`` match first; otherwise the OLDEST still-unresolved
        row for the same ``(tool_name, session_id)`` — FIFO, never LIFO. Hermes
        can fire pre/post with an empty id on one side only, so id-only matching
        would strand rows (the failure mode agent-ledger documented).

        Returns the row id, or 0 when nothing matched (a post without a pre).
        """
        result_bytes = int(result_bytes or 0)
        duration = None if duration_ms is None else int(duration_ms)
        with self._lock:
            row = None
            if tool_call_id:
                row = self._conn.execute(
                    "SELECT id FROM tool_calls WHERE tool_call_id=? AND resolved=0"
                    " ORDER BY id LIMIT 1", (tool_call_id,)
                ).fetchone()
            if row is None:
                # FIFO fallback. When the caller supplied a turn_id, prefer the
                # oldest unresolved row in that turn (tighter match); fall back
                # to the session-wide FIFO so torn-down turns still resolve.
                if turn_id:
                    row = self._conn.execute(
                        "SELECT id FROM tool_calls WHERE resolved=0 AND tool_name=?"
                        " AND session_id=? AND turn_id=?"
                        " ORDER BY id LIMIT 1",
                        (tool_name or "", session_id or "", turn_id or ""),
                    ).fetchone()
                if row is None:
                    row = self._conn.execute(
                        "SELECT id FROM tool_calls WHERE resolved=0 AND tool_name=? AND session_id=?"
                        " ORDER BY id LIMIT 1", (tool_name or "", session_id or "")
                    ).fetchone()
            if row is None:
                cur = self._conn.execute(
                    "INSERT INTO tool_calls (ts, tool_call_id, session_id, tool_name,"
                    " result_bytes, duration_ms, status, resolved)"
                    " VALUES (?,?,?,?,?,?,?,1)",
                    (_now(), tool_call_id or "", session_id or "", tool_name or "",
                     result_bytes, duration, status or ""),
                )
                self._conn.commit()
                return int(cur.lastrowid or 0)
            self._conn.execute(
                "UPDATE tool_calls SET result_bytes=?, duration_ms=?, status=?, resolved=1 WHERE id=?",
                (result_bytes, duration, status or "", row[0]),
            )
            self._conn.commit()
            return int(row[0])

    def add_turn(self, session_id: str = "", turn_id: str = "", model: str = "",
                 platform: str = "", est_input_tokens: int = 0, est_output_tokens: int = 0,
                 largest_tool: str = "", largest_tool_bytes: int = 0,
                 compaction_events: int = 0, ts: float | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns (ts, session_id, turn_id, model, platform, est_input_tokens,"
                " est_output_tokens, largest_tool, largest_tool_bytes, compaction_events)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts if ts is not None else _now(), session_id or "", turn_id or "", model or "",
                 platform or "", int(est_input_tokens or 0), int(est_output_tokens or 0),
                 largest_tool or "", int(largest_tool_bytes or 0), int(compaction_events or 0)),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def add_compaction(self, session_id: str = "", aux_task: str = "compression",
                       model: str = "", ts: float | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO compactions (ts, session_id, aux_task, model) VALUES (?,?,?,?)",
                (ts if ts is not None else _now(), session_id or "", aux_task or "", model or ""),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    # -- reads -----------------------------------------------------------

    def largest_tool_in_turn(self, session_id: str, turn_id: str = "",
                             since_ts: float | None = None) -> tuple[str, int]:
        """(tool_name, bytes) of the biggest tool *output* in the turn.

        Ranks by recorded result size. Falls back to the largest *args* only when
        the turn recorded no result at all (the turn was torn down before any
        ``post_tool_call``), so a killed turn still reports its weight instead of
        reporting nothing.
        """
        with self._lock:
            if turn_id:
                rows = self._conn.execute(
                    "SELECT tool_name, args_bytes, result_bytes FROM tool_calls"
                    " WHERE turn_id=? ORDER BY id", (turn_id,)
                ).fetchall()
            elif since_ts is not None:
                rows = self._conn.execute(
                    "SELECT tool_name, args_bytes, result_bytes FROM tool_calls"
                    " WHERE session_id=? AND ts>=? ORDER BY id", (session_id or "", since_ts)
                ).fetchall()
            else:
                return ("", 0)
        best_name, best_bytes = "", 0
        fallback_name, fallback_bytes = "", 0
        for name, args_bytes, result_bytes in rows:
            name = name or ""
            args_bytes = int(args_bytes or 0)
            result_bytes = int(result_bytes or 0)
            if args_bytes > fallback_bytes:
                fallback_name, fallback_bytes = name, args_bytes
            if result_bytes > best_bytes:
                best_name, best_bytes = name, result_bytes
        if best_bytes:
            return (best_name, best_bytes)
        return (fallback_name, fallback_bytes)

    def compactions_since(self, session_id: str, since_ts: float | None) -> int:
        if since_ts is None:
            return 0
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM compactions WHERE session_id=? AND ts>=?",
                (session_id or "", since_ts),
            ).fetchone()
        return int(row[0] or 0) if row else 0

    def top_eaters(self, since_ts: float, limit: int = _REPORT_LIMIT) -> list[dict]:
        """Rank tools by bytes they pushed into the window over the window."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tool_name,"
                "       COALESCE(SUM(args_bytes + result_bytes), 0) AS total_bytes,"
                "       COUNT(*) AS calls,"
                "       COALESCE(MAX(args_bytes), 0) AS max_args_bytes,"
                "       COALESCE(SUM(oversized), 0) AS oversized_calls,"
                "       COALESCE(SUM(result_bytes), 0) AS total_result_bytes"
                " FROM tool_calls WHERE ts>=? GROUP BY tool_name"
                " ORDER BY total_bytes DESC, calls DESC LIMIT ?",
                (since_ts, int(limit)),
            ).fetchall()
        out = []
        for name, total, calls, max_args, oversized, result_total in rows:
            calls = int(calls or 0)
            total = int(total or 0)
            out.append({
                "tool_name": name or "",
                "total_bytes": total,
                "calls": calls,
                "avg_bytes": int(total / calls) if calls else 0,
                "max_args_bytes": int(max_args or 0),
                "total_result_bytes": int(result_total or 0),
                "oversized_calls": int(oversized or 0),
            })
        return out

    def summary(self, since_ts: float) -> dict:
        with self._lock:
            turns = int(self._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE ts>=?", (since_ts,)).fetchone()[0] or 0)
            calls = int(self._conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE ts>=?", (since_ts,)).fetchone()[0] or 0)
            oversized = int(self._conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE ts>=? AND oversized=1", (since_ts,)).fetchone()[0] or 0)
            compactions = int(self._conn.execute(
                "SELECT COUNT(*) FROM compactions WHERE ts>=?", (since_ts,)).fetchone()[0] or 0)
        return {"turns": turns, "tool_calls": calls, "oversized_calls": oversized,
                "compactions": compactions}

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass


_DB: AuditDB | None = None


def get_db() -> AuditDB:
    """Lazy singleton; recreated when the effective DB path changes.

    The path is resolved lazily at call time so a ``HERMES_HOME`` change after
    import (tests, profile switch) is picked up instead of freezing the path
    computed at module import.
    """
    global _DB
    path = DB_PATH if DB_PATH != _default_db_path() else _default_db_path()
    if _DB is None or _DB.path != path:
        _DB = AuditDB(path)
    return _DB


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

_MITIGATIONS = (
    (re.compile(r"web_extract|browser", re.I),
     "avg {avg} per call — lower char_limit / prefer js()+page_info() over full-page dumps"),
    (re.compile(r"terminal|execute_code|shell", re.I),
     "avg {avg} per call — redirect bulk output to a file and read a slice instead"),
    (re.compile(r"read_file", re.I),
     "avg {avg} per call — pass offset+limit; never paste whole files into context"),
    (re.compile(r"search_files|grep|glob", re.I),
     "avg {avg} per call — narrow the pattern/path; raise no limits you don't need"),
    (re.compile(r"delegate_task|subagent", re.I),
     "avg {avg} per call — send a digest to subagents, not the full transcript"),
    (re.compile(r"skill_view|skill_manage", re.I),
     "avg {avg} per call — load skills lazily; avoid hoisting whole skill bodies"),
)


def _human_bytes(n: int) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def mitigation_for(tool_name: str, avg_bytes: int) -> str:
    """One actionable suggestion for a tool, or a generic fallback."""
    avg = _human_bytes(avg_bytes)
    for pattern, template in _MITIGATIONS:
        if pattern.search(tool_name or ""):
            return template.format(avg=avg)
    return f"avg {avg} per call — trim args/results; this tool is your heaviest context source"


def report(days: int = _WINDOW_DAYS, limit: int = _REPORT_LIMIT, db: AuditDB | None = None) -> dict:
    """Structured report over the trailing ``days`` window."""
    days = max(1, min(int(days or _WINDOW_DAYS), 365))
    since = _now() - days * 86400
    handle = db or get_db()
    return {
        "plugin": PLUGIN_NAME,
        "version": __version__,
        "days": days,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold_bytes": CONTEXT_EATER_BYTES,
        "summary": handle.summary(since),
        "eaters": [
            {**row, "mitigation": mitigation_for(row["tool_name"], row["avg_bytes"])}
            for row in handle.top_eaters(since, limit)
        ],
    }


def render_report(days: int = _WINDOW_DAYS, limit: int = _REPORT_LIMIT,
                  db: AuditDB | None = None) -> str:
    """Human-readable Top-N report — what ``/context-audit`` prints."""
    data = report(days=days, limit=limit, db=db)
    summary = data["summary"]
    lines = [
        f"CONTEXT AUDIT — top {limit} context eaters, last {days} day(s)",
        f"turns={summary['turns']}  tool_calls={summary['tool_calls']}  "
        f"oversized(>{_human_bytes(CONTEXT_EATER_BYTES)} args)={summary['oversized_calls']}  "
        f"compactions={summary['compactions']}",
        "",
    ]
    if not data["eaters"]:
        lines.append("No tool calls recorded in this window.")
        return "\n".join(lines).rstrip()
    for idx, row in enumerate(data["eaters"], 1):
        flag = f"  [{row['oversized_calls']} oversized]" if row["oversized_calls"] else ""
        lines.append(
            f"{idx}. {row['tool_name'] or '(unknown)':<20} {_human_bytes(row['total_bytes']):>9}"
            f"  {row['calls']:>5} calls  avg {_human_bytes(row['avg_bytes'])}{flag}"
        )
        lines.append(f"   -> {row['mitigation']}")
    if summary["compactions"]:
        lines.append("")
        lines.append(f"{summary['compactions']} compaction event(s) in this window — "
                     "a rising count means the window is filling faster than it drains.")
    return "\n".join(lines).rstrip()


def _parse_args(raw: str) -> tuple[int, int, bool]:
    """``raw`` -> (days, limit, as_json). Tolerates junk; never raises."""
    days, limit, as_json = _WINDOW_DAYS, _REPORT_LIMIT, False
    for token in (raw or "").replace(",", " ").split():
        low = token.strip().lower()
        if low in {"json", "--json"}:
            as_json = True
            continue
        if "=" in low:
            key, _, value = low.partition("=")
            if key in {"days", "limit"}:
                try:
                    number = int(value)
                except ValueError:
                    continue
                if key == "days":
                    days = max(1, min(number, 365))
                else:
                    limit = max(1, min(number, 50))
            continue
        try:
            days = max(1, min(int(low), 365))
        except ValueError:
            continue
    return days, limit, as_json


# ---------------------------------------------------------------------------
# hooks — all observer-only, all fail-open
# ---------------------------------------------------------------------------

def _turn_key(session_id: str, turn_id: str) -> tuple[str, str]:
    return (session_id or "", turn_id or "")


def _note_turn_start(session_id: str, turn_id: str) -> float:
    started = _now()
    with _state_lock:
        _turn_state[_turn_key(session_id, turn_id)] = {"started_ts": started}
        if len(_turn_state) > 512:  # bound growth; oldest keys first (dict is ordered)
            for key in list(_turn_state)[: len(_turn_state) - 512]:
                _turn_state.pop(key, None)
    return started


def _turn_started(session_id: str, turn_id: str) -> float | None:
    with _state_lock:
        entry = _turn_state.get(_turn_key(session_id, turn_id))
        return entry.get("started_ts") if entry else None


@_observer
def on_pre_tool_call(tool_name: str = "", args=None, session_id: str = "",
                     tool_call_id: str = "", turn_id: str = "", **kwargs) -> None:
    """Size up the args *before* the call runs; flag the oversized ones."""
    size = _payload_bytes(args)
    get_db().add_tool_call(tool_name, size, session_id=session_id, turn_id=turn_id,
                           tool_call_id=tool_call_id)


@_observer
def on_post_tool_call(tool_name: str = "", args=None, result=None, session_id: str = "",
                      tool_call_id: str = "", turn_id: str = "", duration_ms=None,
                      status: str = "", **kwargs) -> None:
    """Attach the result *size* (never its content) to the existing row."""
    size = len(result) if isinstance(result, str) else _payload_bytes(result)
    get_db().confirm_tool_call(tool_name, session_id=session_id, tool_call_id=tool_call_id,
                               result_bytes=size, duration_ms=duration_ms, status=status,
                               turn_id=turn_id)


@_observer
def on_pre_llm_call(session_id: str = "", turn_id: str = "", conversation_history=None,
                    **kwargs) -> None:
    """Mark the turn start so its window is bounded for later queries."""
    _note_turn_start(session_id, turn_id)


@_observer
def on_post_llm_call(session_id: str = "", turn_id: str = "", conversation_history=None,
                     assistant_response=None, model: str = "", platform: str = "", **kwargs) -> None:
    """One row per turn: token estimates, heaviest tool, compaction count."""
    db = get_db()
    started = _turn_started(session_id, turn_id)
    largest_tool, largest_bytes = db.largest_tool_in_turn(session_id, turn_id, started)
    db.add_turn(
        session_id=session_id, turn_id=turn_id, model=model, platform=platform,
        est_input_tokens=_est_tokens(conversation_history),
        est_output_tokens=_est_tokens(assistant_response),
        largest_tool=largest_tool, largest_tool_bytes=largest_bytes,
        compaction_events=db.compactions_since(session_id, started),
    )


@_observer
def on_post_auxiliary_call(aux_task: str = "", session_id: str = "", model: str = "", **kwargs) -> None:
    """Compaction is an auxiliary LLM call; there is no dedicated hook for it."""
    if (aux_task or "").strip().lower() != "compression":
        return
    get_db().add_compaction(session_id=session_id, aux_task="compression", model=model)


# ---------------------------------------------------------------------------
# slash command
# ---------------------------------------------------------------------------

def context_audit_command(raw_args: str = "") -> str:
    """``/context-audit [days] [limit=N] [json]`` — never raises."""
    try:
        days, limit, as_json = _parse_args(raw_args)
        if as_json:
            return json.dumps(report(days=days, limit=limit), indent=2, ensure_ascii=False)
        return render_report(days=days, limit=limit)
    except Exception as exc:  # noqa: BLE001 - a broken report must not break the session
        _fail_open(exc)
        return f"{PLUGIN_NAME}: report unavailable ({type(exc).__name__}: {exc})"


def register(ctx) -> None:
    """Hermes plugin entry point — hooks are observers, the command is read-only."""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("post_auxiliary_call", on_post_auxiliary_call)
    ctx.register_command(
        "context-audit",
        context_audit_command,
        description="Where the context window went — top context eaters over a window",
        args_hint="[days] [limit=N] [json]",
    )
