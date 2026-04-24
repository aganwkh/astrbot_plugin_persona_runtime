from __future__ import annotations

from .models import BehaviorProbabilityResult, LearningEffect, NormalizedInput, SceneResolution, WeightPatchRecord


class BehaviorProbabilityEngine:
    def build(
        self,
        normalized: NormalizedInput,
        scene_resolution: SceneResolution,
        patches: list[WeightPatchRecord] | None = None,
    ) -> BehaviorProbabilityResult:
        scene = scene_resolution.main_scene
        probabilities = self._base_probabilities(scene)
        reasons = {
            "short_ack": "base scene prior",
            "followup_question": "base scene prior",
            "comfort": "base scene prior",
            "solution": "base scene prior",
        }
        applied_effects: list[LearningEffect] = []

        if scene_resolution.is_feedback and scene_resolution.feedback_target == "followup_question":
            probabilities["followup_question"] = self._clamp(probabilities["followup_question"] - 0.18)
            probabilities["short_ack"] = self._clamp(probabilities["short_ack"] + 0.08)
            reasons["followup_question"] = "feedback indicates follow-up questions should be reduced"

        if any(keyword in normalized.raw_text for keyword in ["累", "难受", "崩溃", "烦", "我服了"]):
            probabilities["comfort"] = self._clamp(probabilities["comfort"] + 0.12)
            reasons["comfort"] = "detected pressure or discomfort cues"

        if scene == "task_request" and any(keyword in normalized.raw_text for keyword in ["代码", "修", "改", "分析", "整理"]):
            probabilities["solution"] = self._clamp(probabilities["solution"] + 0.05)
            reasons["solution"] = "task scene plus explicit execution wording"

        if scene == "status_share":
            probabilities["followup_question"] = self._clamp(probabilities["followup_question"] - 0.05)
            reasons["followup_question"] = "status share defaults to low-pressure continuation"

        for patch in patches or []:
            if patch.patch_type != "behavior_weight_patch" or patch.scene != scene:
                continue
            behavior = patch.target_key
            if behavior not in probabilities:
                continue
            effective_delta = round(patch.delta * patch.decay_factor, 3)
            probabilities[behavior] = self._clamp(probabilities[behavior] + effective_delta)
            probabilities[behavior] = max(patch.min_value, min(probabilities[behavior], patch.max_value))
            reasons[behavior] = f"{reasons.get(behavior, 'behavior prior')}; learning patch #{patch.patch_id or 0} applied"
            applied_effects.append(
                LearningEffect(
                    patch_id=patch.patch_id,
                    patch_type=patch.patch_type,
                    scene=patch.scene,
                    target_key=behavior,
                    delta=effective_delta,
                    reason=patch.reason,
                )
            )

        return BehaviorProbabilityResult(
            scene=scene,
            behavior_probabilities={name: round(value, 3) for name, value in probabilities.items()},
            reasons=reasons,
            applied_effects=applied_effects,
        )

    @staticmethod
    def _base_probabilities(scene: str) -> dict[str, float]:
        if scene == "status_share":
            return {
                "short_ack": 0.82,
                "followup_question": 0.19,
                "comfort": 0.24,
                "solution": 0.05,
            }
        if scene == "complaint":
            return {
                "short_ack": 0.34,
                "followup_question": 0.18,
                "comfort": 0.78,
                "solution": 0.28,
            }
        if scene == "task_request":
            return {
                "short_ack": 0.14,
                "followup_question": 0.33,
                "comfort": 0.07,
                "solution": 0.92,
            }
        return {
            "short_ack": 0.46,
            "followup_question": 0.58,
            "comfort": 0.16,
            "solution": 0.1,
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.05, min(value, 0.95))
