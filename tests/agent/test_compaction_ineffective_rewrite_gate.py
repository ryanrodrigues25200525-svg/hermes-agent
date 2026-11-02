"""Unit contracts for the pre-commit ineffective-rewrite gate (#148103).

A rewrite that still straddles the trigger after helping too little must be
refused before history is rewritten — not discovered after the next
over-threshold response.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from agent.conversation_compression_refusal import _refuse_ineffective_rewrite, _rewrite_cannot_clear


def _messages(chars: int, n: int = 4) -> list:
    each, rest = divmod(chars, n)
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * (each + (1 if i < rest else 0))} for i in range(n)]


def _agent(threshold_tokens: int):
    compressor = SimpleNamespace(threshold_tokens=threshold_tokens, strikes=0)
    compressor.record_rejected_compaction = lambda: setattr(compressor, "strikes", compressor.strikes + 1)
    return SimpleNamespace(
        session_id="sess-test", tools=None, context_compressor=compressor,
        _cached_system_prompt="cached-sp", _emit_warning=lambda *a, **k: None,
    )


def _attempt():
    from agent.conversation_compression import _Attempt

    return _Attempt(snapshot={}, generation=1, started_at=time.monotonic(), seed={"attempt_id": "a1"})


class RewriteCannotClearTests(unittest.TestCase):
    def test_clears_threshold_commits(self) -> None:
        self.assertFalse(_rewrite_cannot_clear(100_000, 80_000, 87_040))

    def test_big_reclaim_over_threshold_commits(self) -> None:
        # 30% reclaimed is fresh progress, not churn, even while still over.
        self.assertFalse(_rewrite_cannot_clear(200_000, 140_000, 87_040))

    def test_tiny_reclaim_over_threshold_refuses(self) -> None:
        # The issue's C2 shape: ~97k -> ~89k against an 87k trigger.
        self.assertTrue(_rewrite_cannot_clear(97_654, 89_246, 87_040))

    def test_missing_numbers_fail_open(self) -> None:
        self.assertFalse(_rewrite_cannot_clear(None, 89_246, 87_040))
        self.assertFalse(_rewrite_cannot_clear(97_654, None, 87_040))
        self.assertFalse(_rewrite_cannot_clear(97_654, 89_246, 0))
        self.assertFalse(_rewrite_cannot_clear(0, 0, 0))


class RefuseIneffectiveRewriteTests(unittest.TestCase):
    def test_refuses_and_strikes(self) -> None:
        agent = _agent(87_040)
        before, after = _messages(97_654), _messages(89_246)
        with mock.patch(
            "agent.conversation_compression_refusal.estimate_request_tokens_rough",
            side_effect=[97_654, 89_246],
        ):
            reason = _refuse_ineffective_rewrite(
                agent, before, after, system_message="", new_system_prompt="",
                attempt=_attempt(),
            )
        self.assertEqual(reason, "cached-sp")
        self.assertEqual(agent.context_compressor.strikes, 1)

    def test_commits_when_clear(self) -> None:
        agent = _agent(87_040)
        with mock.patch(
            "agent.conversation_compression_refusal.estimate_request_tokens_rough",
            side_effect=[97_654, 70_000],
        ):
            self.assertIsNone(
                _refuse_ineffective_rewrite(
                    agent, _messages(97_654), _messages(70_000), system_message="",
                    new_system_prompt="", attempt=_attempt(),
                )
            )
        self.assertEqual(agent.context_compressor.strikes, 0)

    def test_no_threshold_fails_open(self) -> None:
        agent = _agent(0)
        self.assertIsNone(
            _refuse_ineffective_rewrite(
                agent, _messages(97_654), _messages(89_246), system_message="",
                new_system_prompt="", attempt=_attempt(),
            )
        )

    def test_no_snapshot_fails_open(self) -> None:
        agent = _agent(87_040)
        self.assertIsNone(
            _refuse_ineffective_rewrite(
                agent, [], _messages(89_246), system_message="",
                new_system_prompt="", attempt=_attempt(),
            )
        )


if __name__ == "__main__":
    unittest.main()
