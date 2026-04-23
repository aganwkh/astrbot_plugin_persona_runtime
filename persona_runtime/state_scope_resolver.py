from __future__ import annotations

from astrbot.api.event import AstrMessageEvent

from .models import ScopeKeys
from .utils import build_scope_key, safe_get_conversation_id, safe_get_platform_id, safe_get_sender_id


class StateScopeResolver:
    def resolve(self, event: AstrMessageEvent) -> ScopeKeys:
        platform_id = safe_get_platform_id(event)
        bot_id = "astrbot"
        user_id = safe_get_sender_id(event)
        conversation_id = safe_get_conversation_id(event)
        return ScopeKeys(
            user_scope_key=build_scope_key(platform_id, bot_id, user_id),
            session_scope_key=build_scope_key(platform_id, bot_id, conversation_id),
        )
