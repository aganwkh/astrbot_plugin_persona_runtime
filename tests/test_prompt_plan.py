from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin
from persona_runtime.dialogue_policy import DialoguePolicy
from persona_runtime.models import LoreItem, NormalizedInput, SceneResolution
from persona_runtime.prompt_merge_policy import PromptMergePolicy
from persona_runtime.token_budget_guard import TokenBudgetGuard


class PromptPlanTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.merge_policy = PromptMergePolicy(static_persona_prompt="你扮演一个克制自然的角色。")
        self.token_guard = TokenBudgetGuard(max_runtime_chars=800)
        self.policy_builder = DialoguePolicy()

    def _build_plan(self, raw_text: str, scene: str, behavior_probabilities: dict[str, float], lore_items: list[LoreItem]):
        resolution = SceneResolution(
            main_scene=scene,
            scene_scores={
                "status_share": 0.85 if scene == "status_share" else 0.2,
                "casual_chat": 0.75 if scene == "casual_chat" else 0.2,
                "complaint": 0.88 if scene == "complaint" else 0.1,
                "task_request": 0.92 if scene == "task_request" else 0.1,
            },
        )
        behavior_result = type(
            "BehaviorLike",
            (),
            {
                "behavior_probabilities": behavior_probabilities,
            },
        )()
        policy = self.policy_builder.build(NormalizedInput(raw_text=raw_text), resolution, behavior_result)
        plan = self.merge_policy.merge(
            scene_resolution=resolution,
            behavior_result=behavior_result,
            policy=policy,
            lore_items=lore_items,
            selected_examples=[],
            learning_effects=[],
            max_runtime_chars=800,
        )
        return self.token_guard.apply(plan)

    def test_prompt_plan_is_stable_without_lore(self):
        plan = self._build_plan(
            raw_text="刚吃完饭",
            scene="status_share",
            behavior_probabilities={
                "short_ack": 0.82,
                "followup_question": 0.14,
                "comfort": 0.2,
                "solution": 0.05,
            },
            lore_items=[],
        )
        self.assertEqual(plan.main_scene, "status_share")
        self.assertEqual(plan.selected_behavior, "short_ack")
        self.assertNotIn("lore", plan.selected_module_ids())
        self.assertTrue(plan.current_scene_text)
        self.assertTrue(plan.behavior_tendency_text)

    def test_prompt_plan_includes_lore_when_present(self):
        plan = self._build_plan(
            raw_text="聊聊乐队和春日影",
            scene="casual_chat",
            behavior_probabilities={
                "short_ack": 0.4,
                "followup_question": 0.58,
                "comfort": 0.15,
                "solution": 0.1,
            },
            lore_items=[LoreItem(lore_id="mygo_base", text="优先引用用户专属知识库。", score=0.8)],
        )
        self.assertIn("lore", plan.selected_module_ids())
        self.assertEqual(plan.selected_lore_ids, ["mygo_base"])
        self.assertGreater(plan.token_budget.used_chars, 0)

    async def test_prwhy_outputs_key_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, module = build_plugin(Path(tmp), timeout_seconds=1)
            try:
                event = FakeEvent(text="刚吃完饭", user_id="user-1", session_id="room-1")
                req = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req)
                await plugin.on_llm_response(event, module.LLMResponse(text="收到。那先坐一会儿吧。"))
                await plugin.after_message_sent(event)

                prwhy_event = FakeEvent(text="/prwhy", user_id="user-1", session_id="room-1")
                outputs = []
                async for item in plugin.prwhy(prwhy_event):
                    outputs.append(item)
                output = "\n".join(outputs)
                self.assertIn("main_scene: status_share", output)
                self.assertIn("selected_behavior: short_ack", output)
                self.assertIn("behavior_probabilities:", output)
                self.assertIn("selected_modules:", output)
                self.assertIn("token_budget:", output)
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
