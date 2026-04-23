from __future__ import annotations

from .models import NormalizedInput


class EligibilityChecker:
    def is_eligible(self, normalized: NormalizedInput) -> bool:
        if normalized.is_system:
            return False
        if normalized.is_empty and normalized.has_attachment_only:
            return False
        return True
