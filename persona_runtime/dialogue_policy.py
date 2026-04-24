from __future__ import annotations

from .models import BehaviorProbabilityResult, DialoguePolicyResult, NormalizedInput, SceneResolution


class DialoguePolicy:
    _BEHAVIOR_ORDER = {
        "solution": 4,
        "comfort": 3,
        "short_ack": 2,
        "followup_question": 1,
    }

    def build(
        self,
        normalized: NormalizedInput,
        scene_resolution: SceneResolution,
        behavior_result: BehaviorProbabilityResult,
    ) -> DialoguePolicyResult:
        selected_behavior = max(
            behavior_result.behavior_probabilities.items(),
            key=lambda item: (item[1], self._BEHAVIOR_ORDER.get(item[0], 0)),
        )[0]
        main_scene = scene_resolution.main_scene
        need_lore = any(word in normalized.raw_text.casefold() for word in ["mygo", "ave mujica", "春日影", "乐队"])
        reply_length = "medium" if selected_behavior == "solution" else "short"
        followup_intensity = "low" if selected_behavior == "followup_question" else "none"
        allow_followup_question = selected_behavior == "followup_question"
        tone_mode = {
            "status_share": "reserved",
            "casual_chat": "natural",
            "complaint": "gentle",
            "task_request": "steady",
        }.get(main_scene, "flat")
        role_tone_strength = {
            "status_share": 0.72,
            "casual_chat": 0.62,
            "complaint": 0.78,
            "task_request": 0.58,
        }.get(main_scene, 0.65)
        return DialoguePolicyResult(
            selected_behavior=selected_behavior,
            reply_length=reply_length,
            followup_intensity=followup_intensity,
            role_tone_strength=role_tone_strength,
            allow_followup_question=allow_followup_question,
            need_lore_injection=need_lore,
            need_soften_tool_tone=main_scene == "task_request",
            need_task_mode=main_scene == "task_request",
            tone_mode=tone_mode,
            max_followup_question_count=1 if allow_followup_question else 0,
        )
