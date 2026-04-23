from __future__ import annotations

from astrbot.api.event import AstrMessageEvent

from .models import NormalizedInput


class InputNormalizer:
    def normalize(self, event: AstrMessageEvent) -> NormalizedInput:
        raw_text = (getattr(event, "message_str", "") or "").strip()
        streaming = getattr(event, "is_streaming", False)
        if callable(streaming):
            streaming = streaming()
        return NormalizedInput(
            raw_text=raw_text,
            is_empty=(raw_text == ""),
            is_system=False,
            is_command_like=raw_text.startswith("/"),
            has_attachment_only=False,
            is_streaming=bool(streaming or getattr(event, "streaming", False)),
        )
