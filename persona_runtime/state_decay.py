from __future__ import annotations

from .models import SessionState, UserState


class StateDecay:
    def apply(self, user_state: UserState, session_state: SessionState):
        user_state.recent_praise_decay = max(0, user_state.recent_praise_decay - 1)
        user_state.recent_doubt_decay = max(0, user_state.recent_doubt_decay - 1)
        session_state.lore_cooldown = max(0, session_state.lore_cooldown - 1)
        session_state.repeat_tail_count = max(0, session_state.repeat_tail_count - 1)
        return user_state, session_state
