from __future__ import annotations

from .models import LoreItem, MergedPromptPayload


class TokenBudgetGuard:
    def __init__(self, max_runtime_chars: int = 800):
        self.max_runtime_chars = max_runtime_chars

    def apply(self, payload: MergedPromptPayload) -> MergedPromptPayload:
        if self.max_runtime_chars <= 0:
            return MergedPromptPayload(policy_lines=[], lore_items=[], style_lines=[])

        remaining = self.max_runtime_chars
        policy_lines, remaining = self._fit_lines(payload.policy_lines, remaining)

        lore_items: list[LoreItem] = []
        for item in sorted(payload.lore_items, key=lambda x: x.score, reverse=True):
            if remaining <= 0:
                break
            text = self._truncate(item.text, remaining)
            if not text:
                break
            lore_items.append(
                LoreItem(
                    lore_id=item.lore_id,
                    text=text,
                    score=item.score,
                    topic=item.topic,
                )
            )
            remaining -= len(text)

        style_lines, remaining = self._fit_lines(payload.style_lines, remaining)
        return MergedPromptPayload(policy_lines=policy_lines, lore_items=lore_items, style_lines=style_lines)

    def _fit_lines(self, lines: list[str], remaining: int) -> tuple[list[str], int]:
        kept: list[str] = []
        for line in lines:
            if remaining <= 0:
                break
            text = self._truncate(line, remaining)
            if not text:
                break
            kept.append(text)
            remaining -= len(text)
        return kept, remaining

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        suffix = "...<trimmed>"
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)].rstrip() + suffix
