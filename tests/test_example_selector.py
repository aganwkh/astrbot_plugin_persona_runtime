from __future__ import annotations

import unittest

from persona_runtime.example_selector import ExampleSelector
from persona_runtime.models import ExampleRecord, PromptPlan, WeightPatchRecord, NormalizedInput


class ExampleSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = ExampleSelector(max_examples_per_turn=3)

    def test_prefers_scene_and_text_match(self):
        plan = PromptPlan(
            main_scene="status_share",
            scene_scores={"status_share": 0.9, "casual_chat": 0.3, "complaint": 0.1, "task_request": 0.05},
        )
        examples = [
            ExampleRecord(
                example_id=1,
                scene="status_share",
                tags=["brief", "low_pressure"],
                user_message="just finished lunch",
                assistant_reply="nice, sit for a bit first.",
            ),
            ExampleRecord(
                example_id=2,
                scene="task_request",
                tags=["direct"],
                user_message="please review this code",
                assistant_reply="show me the error first.",
            ),
        ]
        selected = self.selector.select(NormalizedInput(raw_text="just finished lunch"), plan, examples)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].example_id, 1)

    def test_applies_example_tag_patch_bonus(self):
        plan = PromptPlan(
            main_scene="casual_chat",
            scene_scores={"status_share": 0.2, "casual_chat": 0.8, "complaint": 0.1, "task_request": 0.05},
        )
        examples = [
            ExampleRecord(
                example_id=1,
                scene="casual_chat",
                tags=["natural"],
                user_message="hello there",
                assistant_reply="hey.",
            ),
            ExampleRecord(
                example_id=2,
                scene="casual_chat",
                tags=["warm"],
                user_message="hello there",
                assistant_reply="good to see you.",
            ),
        ]
        selected = self.selector.select(
            NormalizedInput(raw_text="hello there"),
            plan,
            examples,
            patches=[
                WeightPatchRecord(
                    patch_id=3,
                    patch_type="example_tag_weight_patch",
                    scene="casual_chat",
                    target_key="warm",
                    delta=0.2,
                    min_value=-0.4,
                    max_value=0.4,
                    reason="warm examples worked recently",
                )
            ],
        )
        self.assertEqual(selected[0].example_id, 2)
        self.assertEqual(selected[0].applied_effects[0].patch_id, 3)


if __name__ == "__main__":
    unittest.main()
