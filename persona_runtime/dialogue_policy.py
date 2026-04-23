from __future__ import annotations

from .models import DialoguePolicyResult, NormalizedInput, ResolutionResult


class DialoguePolicy:
    def build(self, normalized: NormalizedInput, resolution: ResolutionResult) -> DialoguePolicyResult:
        scores = resolution.feature_scores
        policy = DialoguePolicyResult(
            reply_mode=resolution.preliminary_reply_mode,
            tone_mode=resolution.preliminary_mood,
            allow_followup_question=False,
            need_lore_injection=any(word in normalized.raw_text.lower() for word in ["mygo", "ave mujica", "春日影", "乐队"]),
            need_soften_tool_tone=scores.get("task_score", 0) > 0,
            need_task_mode=scores.get("task_score", 0) > 0,
            max_followup_question_count=0,
        )
        if scores.get("status_share_score", 0) <= 0 and scores.get("task_score", 0) <= 0:
            policy.allow_followup_question = True
            policy.max_followup_question_count = 1
        return policy
