from __future__ import annotations

from dataclasses import dataclass

from .observability_logger import ObservabilityLogger


@dataclass(frozen=True)
class FallbackDecision:
    stage: str
    should_abort_turn: bool = True
    should_skip_runtime: bool = True
    reason: str = ""


class FallbackController:
    def handle(
        self,
        exc: Exception,
        stage: str,
        logger: ObservabilityLogger,
    ) -> FallbackDecision:
        decision = FallbackDecision(
            stage=stage,
            should_abort_turn=stage in {"on_llm_request", "after_message_sent"},
            should_skip_runtime=stage == "on_llm_request",
            reason=type(exc).__name__,
        )
        logger.exception(
            "fallback",
            exc,
            stage=stage,
            fallback_triggered=True,
            decision=decision.__dict__,
        )
        return decision
