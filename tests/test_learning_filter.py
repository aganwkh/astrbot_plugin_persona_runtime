from __future__ import annotations

import unittest

from persona_runtime.learning_filter import LearningFilter
from persona_runtime.models import (
    BehaviorProbabilityResult,
    DialoguePolicyResult,
    NormalizedInput,
    PromptPlan,
    RuntimeContextBundle,
    SceneResolution,
    TokenBudgetSummary,
)


class LearningFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = LearningFilter()

    def _bundle(self, *, user_text: str, assistant_text: str, scene: str = "casual_chat", is_feedback: bool = False, feedback_target: str = "none"):
        return RuntimeContextBundle(
            turn_id="turn-1",
            user_scope_key="user:1",
            session_scope_key="session:1",
            normalized_input=NormalizedInput(raw_text=user_text),
            feature_scores={},
            scene_resolution=SceneResolution(
                main_scene=scene,
                scene_scores={scene: 0.9},
                is_feedback=is_feedback,
                feedback_target=feedback_target,
            ),
            behavior_result=BehaviorProbabilityResult(scene=scene, behavior_probabilities={"short_ack": 0.6}),
            policy=DialoguePolicyResult(selected_behavior="short_ack"),
            prompt_plan=PromptPlan(
                main_scene=scene,
                scene_scores={scene: 0.9},
                selected_behavior="short_ack",
                behavior_probabilities={"short_ack": 0.6},
                token_budget=TokenBudgetSummary(max_runtime_chars=800),
            ),
            selected_lore_ids=[],
            injected_len=120,
            assistant_reply_text=assistant_text,
            unified_msg_origin="dev:room:user",
        )

    def test_accepts_short_chat_turn(self):
        item = self.filter.build_item(self._bundle(user_text="刚吃完饭", assistant_text="嗯。那先坐一会儿吧。", scene="status_share"))
        self.assertIsNotNone(item)
        self.assertEqual(item.content_mode, "full_short")
        self.assertEqual(item.scene, "status_share")

    def test_keeps_feedback_excerpt_for_long_turn(self):
        long_user = "这段代码不对 " * 80
        long_reply = "我先解释一下 " * 80
        item = self.filter.build_item(
            self._bundle(
                user_text=long_user,
                assistant_text=long_reply,
                scene="complaint",
                is_feedback=True,
                feedback_target="accuracy",
            )
        )
        self.assertIsNotNone(item)
        self.assertEqual(item.content_mode, "feedback_excerpt")
        self.assertEqual(item.feedback_label, "accuracy")

    def test_skips_code_heavy_turn_without_feedback(self):
        code = "```python\n" + ("print('x')\n" * 80) + "```"
        item = self.filter.build_item(self._bundle(user_text=code, assistant_text="先看报错位置。", scene="task_request"))
        self.assertIsNone(item)


if __name__ == "__main__":
    unittest.main()
