"""Flush answers "already durable?" from the database, not from memory (#148114).

Rebuilt message lists (compaction, retried turn, resumed session) share no
object identity with the previous flush snapshot and carry no markers, so the
list used to read as "all new" and every row was inserted again with its
original timestamp. The durable-key probe -- ``(timestamp, role,
tool_call_id)`` already stored, longest content wins -- closes all three
flush paths at once with no call-site changes.
"""

import copy

from agent.message_metadata import TOOL_CALL_UID
from agent.session_persistence import _db_flush_collect
from hermes_state import SessionDB

_T0 = 1700000000.0


def _turn(prefix=""):
    return [
        {"role": "user", "content": f"{prefix}question", "timestamp": _T0},
        {
            "role": "assistant",
            "content": f"{prefix}let me check",
            "tool_calls": [{"id": "call_1", "name": "terminal", "arguments": "{}"}],
            "timestamp": _T0 + 1.0,
        },
        {
            "role": "tool",
            "content": f"{prefix}{'x' * 5000}",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
            TOOL_CALL_UID: "tuid-1",
            "timestamp": _T0 + 2.0,
        },
    ]


def _agent(db, session_id):
    from types import SimpleNamespace

    return SimpleNamespace(session_id=session_id, _session_db=db, _last_flushed_db_idx=0)


def _stored_count(db, session_id):
    return len(db.get_messages(session_id))


class TestRebuiltListFlushesNothing:
    def test_fresh_dicts_with_same_timestamps_write_no_rows(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())
            assert _stored_count(db, "s") == 3
            rebuilt = copy.deepcopy(_turn())
            rows, _ = _db_flush_collect(_agent(db, "s"), rebuilt, None)
            assert rows == []
            second, _ = _db_flush_collect(_agent(db, "s"), rebuilt, None)
            assert second == []
            assert _stored_count(db, "s") == 3


class TestDemotedStubRejected:
    def test_shorter_body_under_existing_key_is_dropped(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())
            stub = copy.deepcopy(_turn())
            stub[2]["content"] = "demoted stub"
            rows, _ = _db_flush_collect(_agent(db, "s"), stub, None)
            assert rows == []
            assert _stored_count(db, "s") == 3
            stored = db.get_messages("s")
            assert len(stored[2]["content"]) == 5000

    def test_richer_later_body_is_written(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            short = _turn()
            short[2]["content"] = "demoted stub"
            db.append_messages_batch("s", short)
            richer = copy.deepcopy(_turn())
            rows, _ = _db_flush_collect(_agent(db, "s"), richer, None)
            assert len(rows) == 1
            assert rows[0]["content"] == "x" * 5000

    def test_new_timestamp_is_written(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())
            fresh = [{"role": "user", "content": "brand new", "timestamp": _T0 + 100.0}]
            rows, _ = _db_flush_collect(_agent(db, "s"), fresh, None)
            assert len(rows) == 1


class TestReplayHistoryStillRewrites:
    def test_replay_skips_the_probe(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())
            rebuilt = copy.deepcopy(_turn())
            rows, _ = _db_flush_collect(_agent(db, "s"), rebuilt, None, replay_history=True)
            assert len(rows) == 3


class TestProbeOutageFailsOpen:
    def test_no_db_handle_writes_everything(self):
        from types import SimpleNamespace

        agent = SimpleNamespace(session_id="s", _session_db=None, _last_flushed_db_idx=0)
        rows, _ = _db_flush_collect(agent, copy.deepcopy(_turn()), None)
        assert len(rows) == 3

    def test_probe_error_writes_everything(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())

            def _boom(_session_id):
                raise RuntimeError("store unreachable")

            db.conversation_keys = _boom
            rows, _ = _db_flush_collect(_agent(db, "s"), copy.deepcopy(_turn()), None)
            assert len(rows) == 3

    def test_probe_runs_once_per_flush(self, tmp_path):
        with SessionDB(tmp_path / "state.db") as db:
            db.create_session("s", source="cli")
            db.append_messages_batch("s", _turn())
            calls = []
            orig = db.conversation_keys
            db.conversation_keys = lambda sid: (calls.append(sid), orig(sid))[1]
            rows, _ = _db_flush_collect(_agent(db, "s"), copy.deepcopy(_turn()), None)
            assert rows == []
            assert calls == ["s"]


if __name__ == "__main__":
    import unittest

    unittest.main()
