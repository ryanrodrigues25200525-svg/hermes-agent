"""Commit-site refusal guards for compaction (moved from the facade).

``_salvage_or_refuse_grown_transcript`` runs inside ``_commit_compaction``
before any session mutation: the anti-growth guard plus the
still-over-threshold guard. Both refuse as a no-op (the original transcript
stays untouched and durable) with an ineffective-compaction strike so the
anti-thrash breaker latches instead of retrying the identical summary every
turn. Late facade imports inside the function follow the sibling rule: the
facade owns those names.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

from agent.conversation_compression_telemetry import _emit_aborted_attempt_telemetry
from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough

logger = logging.getLogger("agent.conversation_compression")

# A rewrite that clears the trigger always commits; so does one that reclaims
# at least this share even while staying over (fresh progress, not churn).
_MIN_REWRITE_RECLAIM_FRACTION = 0.10


def _rewrite_cannot_clear(pre_est: int | None, post_est: int | None, threshold: int | None) -> bool:
    """True when a rewrite still straddles the trigger after helping too little.

    Committing it pays the rewrite + re-prefill churn and cascades another
    compression one turn later, for a transcript that still cannot clear the
    trigger. Missing numbers fail open to today's behavior.
    """
    if not pre_est or not post_est or not threshold:
        return False
    if post_est < threshold:
        return False
    return (pre_est - post_est) / pre_est < _MIN_REWRITE_RECLAIM_FRACTION


def _salvage_or_refuse_grown_transcript(
    agent: Any, original_messages: list, compressed: list, *, system_message: str,
    new_system_prompt: str = "", attempt: Any = None,
) -> tuple[Optional[list], Optional[str]]:
    """Anti-growth guard at the COMMIT SITE (in-place commits before the gateway can inspect).
    Compares like-for-like rough estimates; on growth tries one mechanical salvage pass, else treats the
    attempt as a refused no-op. A rewrite that survives growth but still cannot clear the trigger after
    helping too little is refused the same way. Returns ``(compressed, None)`` to proceed or
    ``(None, prompt)`` when refused (caller releases the lease)."""
    # Anti-growth guard at the COMMIT SITE: never persist a compression that makes the transcript larger
    # (observed: 379K -> 687K when the generated summary plus retained reasoning exceeded what it replaced).
    # Compare like-for-like (both rough estimates of the same message shape) so an "actual vs estimate"
    # measurement mismatch cannot produce a false verdict. The gateway has a rotation-path-only guard
    # (#83339), but in-place compaction commits inside this method via archive_and_compact — before the
    # gateway can inspect the result — so the guard must live here to protect both paths. On growth, treat
    # the attempt as a no-op: the original transcript stays untouched and durable.
    _rough_in = estimate_messages_tokens_rough(original_messages)
    _rough_out = estimate_messages_tokens_rough(compressed)
    if _rough_out > _rough_in:
        # Todo refresh and user-turn anchoring run after the compressor's own size check
        # and can tip a break-even candidate; give it one mechanical salvage pass.
        from agent.context_compressor import salvage_grown_transcript
        _salvaged = salvage_grown_transcript(original_messages, compressed, budget=_rough_in)
        if _salvaged is not None:
            _salv_est = estimate_messages_tokens_rough(_salvaged)
            if _salv_est < _rough_in:
                logger.info(
                    "Compression salvage recovered a shrinking transcript (session=%s, ~%s -> ~%s tokens)",
                    agent.session_id or "none", f"{_rough_in:,}", f"{_salv_est:,}",
                )
                compressed = _salvaged
                _rough_out = _salv_est
    if _rough_out > _rough_in:
        logger.warning(
            "Compression refused: compressed transcript would be larger than the original (session=%s, ~%s -> ~%s "
            "tokens); keeping the original transcript unchanged", agent.session_id or "none",
            f"{_rough_in:,}",
            f"{_rough_out:,}",
        )
        # Flag the refusal on compressor state so /compress feedback reports it instead
        # of comparing list lengths (adoption can change the count), claiming success.
        with contextlib.suppress(Exception):
            agent.context_compressor._last_compress_refused_would_grow = True
        with contextlib.suppress(Exception):
            agent._emit_warning(
                "⚠️ Compression refused: the generated summary would have GROWN the conversation instead of "
                "shrinking it. No messages were dropped — conversation continues unchanged."
            )
        from agent.conversation_compression import _existing_system_prompt
        _existing_sp = _existing_system_prompt(agent, system_message)
        _emit_aborted_attempt_telemetry(agent, attempt.started_at, "would_grow", attempt.seed)
        # Count the refusal as an ineffective-compaction strike so the anti-thrash
        # breaker latches; otherwise auto-compress retries the same summary every turn.
        with contextlib.suppress(Exception):
            # Without this, the unchanged transcript stays over the compression threshold and automatic
            # compression retries the identical summary request on every turn (#88568). Manual /compress
            # keeps bypassing the latch (force=True skips the guards).
            agent.context_compressor.record_rejected_compaction()
        _restore_prune_tokens(agent, attempt)
        return None, _existing_sp
    _refused_prompt = _refuse_ineffective_rewrite(
        agent, original_messages, compressed, system_message=system_message,
        new_system_prompt=new_system_prompt, attempt=attempt,
    )
    if _refused_prompt is not None:
        return None, _refused_prompt
    return compressed, None


def _refuse_ineffective_rewrite(agent: Any, original_messages: list, compressed: list, *,
                                system_message: str, new_system_prompt: str, attempt: Any) -> Optional[str]:
    """The existing system prompt when the rewrite cannot clear the trigger, else None.

    Full-request estimates on both sides (same estimator, apples to apples):
    committing a rewrite that still straddles the trigger after reclaiming
    almost nothing pays rewrite + re-prefill churn and cascades another
    compression one turn later (#148103). Refusing records the ineffective
    strike immediately through the pre-commit rejection instead of discovering
    it after the next over-threshold response. Missing numbers fail open.
    """
    from agent.conversation_compression import _existing_system_prompt
    compressor = agent.context_compressor
    threshold = int(getattr(compressor, "threshold_tokens", 0) or 0)
    if not threshold or attempt is None:
        return None
    tools = getattr(agent, "tools", None)
    pre_est = estimate_request_tokens_rough(original_messages, system_prompt=system_message or "", tools=tools)
    post_est = estimate_request_tokens_rough(
        compressed, system_prompt=(new_system_prompt or system_message) or "", tools=tools)
    if not _rewrite_cannot_clear(pre_est, post_est, threshold):
        return None
    logger.warning(
        "Compression refused: rewrite cannot clear the trigger (session=%s, rough ~%s->~%s vs threshold %s, "
        "reclaims %.0f%%); keeping the original transcript unchanged",
        agent.session_id or "none", f"{pre_est:,}", f"{post_est:,}", f"{threshold:,}",
        (pre_est - post_est) / pre_est,
    )
    _emit_aborted_attempt_telemetry(agent, attempt.started_at, "estimate_still_over_threshold", attempt.seed)
    with contextlib.suppress(Exception):
        compressor.record_rejected_compaction()
    _restore_prune_tokens(agent, attempt)
    return _existing_system_prompt(agent, system_message)


def _restore_prune_tokens(agent: Any, attempt: Any) -> None:
    """Restore the prune-rearm tokens the attempt snapshot holds."""
    try:
        from agent.conversation_compression import _restore_prune_rearm_tokens
        _restore_prune_rearm_tokens(agent.context_compressor, attempt.snapshot)
    except Exception:
        logger.debug("failed to restore prune rearm tokens", exc_info=True)
