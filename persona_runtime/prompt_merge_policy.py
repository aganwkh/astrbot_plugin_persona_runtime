from __future__ import annotations

from .models import (
    BehaviorProbabilityResult,
    DialoguePolicyResult,
    ExampleSelection,
    LearningEffect,
    LoreItem,
    PromptModuleSelection,
    PromptPlan,
    SceneResolution,
    TokenBudgetSummary,
)


class PromptMergePolicy:
    def __init__(self, static_persona_prompt: str = ""):
        self.static_persona_prompt = (static_persona_prompt or "").strip()

    def merge(
        self,
        scene_resolution: SceneResolution,
        behavior_result: BehaviorProbabilityResult,
        policy: DialoguePolicyResult,
        lore_items: list[LoreItem],
        selected_examples: list[ExampleSelection],
        learning_effects: list[LearningEffect],
        max_runtime_chars: int,
    ) -> PromptPlan:
        modules: list[PromptModuleSelection] = []
        debug_reasons = [
            f"scene={scene_resolution.main_scene}: {scene_resolution.reasons.get(scene_resolution.main_scene, 'scene heuristic selected')}",
            f"behavior={policy.selected_behavior}: selected by max-weight policy for the scene",
        ]

        if self.static_persona_prompt:
            modules.append(
                PromptModuleSelection(
                    module_id="base_persona",
                    title="Core Persona",
                    content=self.static_persona_prompt,
                    score=1.0,
                    reason="static_persona_prompt configured",
                )
            )

        style_lines = [
            "短句优先，先接住用户当前这句话。",
            "语气保持自然和克制，避免客服式收尾。",
        ]
        if policy.need_soften_tool_tone:
            style_lines.append("处理任务请求时先给结论，再补必要细节，不要写成系统汇报。")
        modules.append(
            PromptModuleSelection(
                module_id="base_style",
                title="Base Style",
                content="\n".join(style_lines),
                score=1.0,
                reason="always-on base style",
            )
        )

        if lore_items:
            modules.append(
                PromptModuleSelection(
                    module_id="lore",
                    title="Relevant Lore",
                    content="\n".join(item.text for item in lore_items),
                    score=max(item.score for item in lore_items),
                    reason="keyword-matched lore for this turn",
                )
            )
            debug_reasons.append(
                "lore matched: " + ", ".join(item.lore_id for item in lore_items)
            )

        if selected_examples:
            debug_reasons.append(
                "examples selected: " + ", ".join(f"#{example.example_id}" for example in selected_examples)
            )
        if learning_effects:
            debug_reasons.append(
                "learning effects: " + ", ".join(f"#{effect.patch_id}:{effect.target_key}" for effect in learning_effects)
            )

        negative_pattern_text = self._negative_pattern_text(scene_resolution.main_scene, policy.selected_behavior)
        if negative_pattern_text:
            modules.append(
                PromptModuleSelection(
                    module_id="negative_pattern_reminder",
                    title="Negative Pattern Reminder",
                    content=negative_pattern_text,
                    score=0.8,
                    reason="scene-specific guardrail",
                )
            )

        return PromptPlan(
            main_scene=scene_resolution.main_scene,
            scene_scores=scene_resolution.scene_scores,
            is_feedback=scene_resolution.is_feedback,
            feedback_target=scene_resolution.feedback_target,
            behavior_probabilities=behavior_result.behavior_probabilities,
            selected_behavior=policy.selected_behavior,
            reply_length=policy.reply_length,
            followup_intensity=policy.followup_intensity,
            role_tone_strength=policy.role_tone_strength,
            current_scene_text=self._scene_text(scene_resolution.main_scene, scene_resolution.is_feedback),
            behavior_tendency_text=self._behavior_tendency_text(policy),
            selected_examples=selected_examples,
            selected_modules=modules,
            selected_lore_ids=[item.lore_id for item in lore_items],
            learning_effects=learning_effects,
            debug_reasons=debug_reasons,
            token_budget=TokenBudgetSummary(max_runtime_chars=max_runtime_chars),
        )

    @staticmethod
    def _scene_text(main_scene: str, is_feedback: bool) -> str:
        scene_texts = {
            "status_share": "当前更像是在分享近况，重点是轻量承接，不要把一句近况拉成审问式对话。",
            "casual_chat": "当前是普通闲聊，允许自然接话，但不要抢着主导话题。",
            "complaint": "当前更像是在吐槽或表达不适，先低压接住情绪，不急着分析。",
            "task_request": "当前是明确请求帮忙处理事情，优先完成任务本身。",
        }
        text = scene_texts.get(main_scene, "当前是普通对话，先回应用户眼前这句话。")
        if is_feedback:
            text += " 这句话还带有对上一轮回复的反馈，需要避免重复犯同类问题。"
        return text

    @staticmethod
    def _behavior_tendency_text(policy: DialoguePolicyResult) -> str:
        if policy.selected_behavior == "short_ack":
            return "本轮优先短句承接，保持自然和收敛，不主动追问。"
        if policy.selected_behavior == "comfort":
            return "本轮优先给低压的关心或安抚，先接情绪，再决定是否延续话题。"
        if policy.selected_behavior == "solution":
            return "本轮优先直接给出可执行结论，再补最必要的说明。"
        return "本轮允许追问，但只问一个低压问题，不能连续追问。"

    @staticmethod
    def _negative_pattern_text(main_scene: str, selected_behavior: str) -> str:
        if main_scene == "status_share":
            return "避免连续追问，尤其不要把近况分享改写成多问句盘问。"
        if main_scene == "complaint":
            return "避免机械总结情绪或立刻讲大道理，先低压回应。"
        if main_scene == "task_request" and selected_behavior == "solution":
            return "避免先铺陈客套或自我说明，直接进入处理结果。"
        return ""
