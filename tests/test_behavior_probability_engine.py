from __future__ import annotations

import unittest

from persona_runtime.behavior_probability_engine import BehaviorProbabilityEngine
from persona_runtime.models import NormalizedInput, SceneResolution, WeightPatchRecord


class BehaviorProbabilityEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = BehaviorProbabilityEngine()

    def test_status_share_prefers_short_ack(self):
        resolution = SceneResolution(
            main_scene="status_share",
            scene_scores={"status_share": 0.9, "casual_chat": 0.4, "complaint": 0.1, "task_request": 0.05},
        )
        result = self.engine.build(NormalizedInput(raw_text="just finished lunch"), resolution)
        self.assertGreater(result.behavior_probabilities["short_ack"], result.behavior_probabilities["followup_question"])

    def test_complaint_prefers_comfort(self):
        resolution = SceneResolution(
            main_scene="complaint",
            scene_scores={"status_share": 0.1, "casual_chat": 0.2, "complaint": 0.9, "task_request": 0.05},
        )
        result = self.engine.build(NormalizedInput(raw_text="this is rough"), resolution)
        self.assertGreater(result.behavior_probabilities["comfort"], result.behavior_probabilities["short_ack"])
        self.assertGreater(result.behavior_probabilities["comfort"], result.behavior_probabilities["solution"])

    def test_task_request_prefers_solution(self):
        resolution = SceneResolution(
            main_scene="task_request",
            scene_scores={"status_share": 0.05, "casual_chat": 0.2, "complaint": 0.1, "task_request": 0.95},
        )
        result = self.engine.build(NormalizedInput(raw_text="please review this code"), resolution)
        self.assertGreater(result.behavior_probabilities["solution"], result.behavior_probabilities["followup_question"])

    def test_applies_behavior_learning_patch(self):
        resolution = SceneResolution(
            main_scene="casual_chat",
            scene_scores={"status_share": 0.1, "casual_chat": 0.8, "complaint": 0.2, "task_request": 0.05},
        )
        result = self.engine.build(
            NormalizedInput(raw_text="hello there"),
            resolution,
            patches=[
                WeightPatchRecord(
                    patch_id=7,
                    patch_type="behavior_weight_patch",
                    scene="casual_chat",
                    target_key="comfort",
                    delta=0.12,
                    reason="test patch",
                )
            ],
        )
        self.assertGreater(result.behavior_probabilities["comfort"], 0.16)
        self.assertEqual(len(result.applied_effects), 1)
        self.assertEqual(result.applied_effects[0].patch_id, 7)


if __name__ == "__main__":
    unittest.main()
