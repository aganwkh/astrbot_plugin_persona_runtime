from __future__ import annotations

from .models import MergedPromptPayload


class PromptComposer:
    def __init__(self, static_persona_prompt: str = ""):
        self.static_persona_prompt = (static_persona_prompt or "").strip()

    def compose(self, payload: MergedPromptPayload) -> str:
        parts: list[str] = []
        if self.static_persona_prompt:
            parts.append(self.static_persona_prompt)
        parts.append("Current-turn runtime rules:")
        for line in payload.policy_lines:
            parts.append(f"- {line}")
        if payload.lore_items:
            parts.append("Current-turn relevant lore:")
            for item in payload.lore_items:
                parts.append(f"- {item.text}")
        if payload.style_lines:
            parts.append("Current-turn style guidance:")
            for line in payload.style_lines:
                parts.append(f"- {line}")
        return "\n".join(parts).strip()
