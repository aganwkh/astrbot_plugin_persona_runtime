from __future__ import annotations

from .models import NormalizedInput, ResolutionResult, SessionState, UserState


class StateResolver:
    def resolve(self, normalized: NormalizedInput, user_state: UserState, session_state: SessionState) -> ResolutionResult:
        text = normalized.raw_text
        praise_score = 1.0 if any(x in text for x in ["厉害", "靠谱", "能干", "好棒", "牛"]) else 0.0
        doubt_score = 1.0 if any(x in text for x in ["你会不会", "不行", "不靠谱", "有问题"]) else 0.0
        status_share_score = 1.0 if any(x in text for x in ["吃完饭", "刚下课", "到家了", "在忙", "睡觉"]) else 0.0
        task_score = 1.0 if any(x in text for x in ["帮我", "你看看", "查一下", "分析", "整理"]) else 0.0

        mood = "flat"
        if praise_score > 0:
            mood = "bright"
        elif doubt_score > 0:
            mood = "defensive"

        reply_mode = "task" if task_score > 0 else "short" if status_share_score > 0 else "normal"
        return ResolutionResult(
            feature_scores={
                "praise_score": praise_score,
                "doubt_score": doubt_score,
                "status_share_score": status_share_score,
                "task_score": task_score,
            },
            preliminary_mood=mood,
            preliminary_reply_mode=reply_mode,
            pre_turn_flags={"status_share": status_share_score > 0, "task_like": task_score > 0},
        )
