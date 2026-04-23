from __future__ import annotations

from .models import BypassDecision, NormalizedInput


class BypassRouter:
    def decide(self, normalized: NormalizedInput) -> BypassDecision:
        if normalized.is_command_like:
            return BypassDecision(True, "command")
        return BypassDecision(False, "")
