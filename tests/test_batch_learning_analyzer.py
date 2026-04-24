from __future__ import annotations

import unittest

from persona_runtime.batch_learning_analyzer import BatchLearningAnalyzer
from persona_runtime.models import ExampleRecord, LearningBufferItem, TurnTraceRecord


class BatchLearningAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = BatchLearningAnalyzer(min_batch_size=3)

    def test_generates_behavior_and_example_tag_patches(self):
        items = [
            LearningBufferItem(
                buffer_id=1,
                turn_id="turn-1",
                user_excerpt="just finished lunch",
                assistant_excerpt="nice, sit for a bit first.",
                scene="status_share",
                selected_behavior="short_ack",
            ),
            LearningBufferItem(
                buffer_id=2,
                turn_id="turn-2",
                user_excerpt="just got home",
                assistant_excerpt="rest first.",
                scene="status_share",
                selected_behavior="short_ack",
            ),
            LearningBufferItem(
                buffer_id=3,
                turn_id="turn-3",
                user_excerpt="done with class",
                assistant_excerpt="take a break.",
                scene="status_share",
                selected_behavior="short_ack",
            ),
        ]
        traces = {
            "turn-1": TurnTraceRecord(
                turn_id="turn-1",
                scene="status_share",
                selected_behavior="short_ack",
                behavior_probabilities={"short_ack": 0.8},
                scene_scores={"status_share": 0.9},
                selected_example_ids=[1],
                selected_lore_ids=[],
            ),
            "turn-2": TurnTraceRecord(
                turn_id="turn-2",
                scene="status_share",
                selected_behavior="short_ack",
                behavior_probabilities={"short_ack": 0.8},
                scene_scores={"status_share": 0.9},
                selected_example_ids=[1],
                selected_lore_ids=[],
            ),
            "turn-3": TurnTraceRecord(
                turn_id="turn-3",
                scene="status_share",
                selected_behavior="short_ack",
                behavior_probabilities={"short_ack": 0.8},
                scene_scores={"status_share": 0.9},
                selected_example_ids=[1],
                selected_lore_ids=[],
            ),
        }
        examples = {
            1: ExampleRecord(
                example_id=1,
                turn_id="turn-0",
                scene="status_share",
                tags=["brief"],
                user_message="just finished lunch",
                assistant_reply="nice, sit for a bit first.",
            )
        }

        patches = self.analyzer.analyze(items, traces, examples)
        patch_targets = {(patch.patch_type, patch.target_key) for patch in patches}

        self.assertIn(("behavior_weight_patch", "short_ack"), patch_targets)
        self.assertIn(("example_tag_weight_patch", "brief"), patch_targets)

    def test_feedback_turns_reduce_followup_question(self):
        items = [
            LearningBufferItem(
                buffer_id=1,
                turn_id="turn-1",
                user_excerpt="stop asking me that",
                assistant_excerpt="ok",
                scene="complaint",
                selected_behavior="followup_question",
                feedback_label="followup_question",
            ),
            LearningBufferItem(
                buffer_id=2,
                turn_id="turn-2",
                user_excerpt="please do not keep asking",
                assistant_excerpt="understood",
                scene="complaint",
                selected_behavior="followup_question",
                feedback_label="followup_question",
            ),
            LearningBufferItem(
                buffer_id=3,
                turn_id="turn-3",
                user_excerpt="you keep pressing",
                assistant_excerpt="got it",
                scene="complaint",
                selected_behavior="followup_question",
                feedback_label="followup_question",
            ),
        ]

        patches = self.analyzer.analyze(items, traces_by_turn={}, examples_by_id={})
        negative_patch = next(
            patch for patch in patches if patch.patch_type == "behavior_weight_patch" and patch.target_key == "followup_question"
        )

        self.assertLess(negative_patch.delta, 0.0)
        self.assertEqual(negative_patch.scene, "complaint")


if __name__ == "__main__":
    unittest.main()
