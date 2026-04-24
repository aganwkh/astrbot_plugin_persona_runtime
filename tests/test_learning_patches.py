from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin
from persona_runtime.models import ExampleRecord, LearningBufferItem, TurnTraceRecord, WeightPatchRecord


class LearningPatchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_prlearn_persists_patches_from_pending_buffer(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, _module = build_plugin(Path(tmp), timeout_seconds=1)
            try:
                await plugin._ensure_ready()
                example = await plugin.repository.create_example(
                    ExampleRecord(
                        turn_id="seed-turn",
                        scene="status_share",
                        tags=["brief"],
                        user_message="just finished lunch",
                        assistant_reply="nice, sit for a bit first.",
                    )
                )
                for index in range(3):
                    turn_id = f"turn-{index}"
                    await plugin.repository.create_turn_trace(
                        TurnTraceRecord(
                            turn_id=turn_id,
                            scene="status_share",
                            selected_behavior="short_ack",
                            behavior_probabilities={"short_ack": 0.82},
                            scene_scores={"status_share": 0.9},
                            selected_example_ids=[int(example.example_id or 0)],
                            selected_lore_ids=[],
                        )
                    )
                    await plugin.repository.create_learning_buffer_item(
                        LearningBufferItem(
                            turn_id=turn_id,
                            user_excerpt=f"status {index}",
                            assistant_excerpt="ok",
                            scene="status_share",
                            selected_behavior="short_ack",
                        )
                    )

                event = FakeEvent(text="/prlearn", user_id="user-1", session_id="room-1")
                outputs = []
                async for item in plugin.prlearn(event):
                    outputs.append(item)
                output = "\n".join(outputs)

                patches = await plugin.repository.list_weight_patches(limit=10)
                pending = await plugin.repository.list_pending_learning_buffer(limit=10)

                self.assertIn("created_patches:", output)
                self.assertGreaterEqual(len(patches), 2)
                self.assertEqual(pending, [])
            finally:
                await plugin.terminate()

    async def test_prwhy_shows_applied_learning_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, module = build_plugin(Path(tmp), timeout_seconds=1)
            try:
                await plugin._ensure_ready()
                await plugin.repository.create_weight_patch(
                    WeightPatchRecord(
                        patch_type="behavior_weight_patch",
                        scene="casual_chat",
                        target_key="short_ack",
                        delta=0.3,
                        evidence_count=3,
                        reason="recent casual chat turns preferred short acknowledgements",
                    )
                )

                event = FakeEvent(text="hello there", user_id="user-1", session_id="room-1")
                req = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req)

                turn_id = plugin.turn_registry.find_latest_turn_id(event)
                turn = plugin.turn_registry.get(turn_id)
                self.assertIsNotNone(turn)
                self.assertEqual(turn.bundle.prompt_plan.selected_behavior, "short_ack")
                self.assertTrue(turn.bundle.prompt_plan.learning_effects)

                why_event = FakeEvent(text="/prwhy", user_id="user-1", session_id="room-1")
                outputs = []
                async for item in plugin.prwhy(why_event):
                    outputs.append(item)
                output = "\n".join(outputs)

                self.assertIn("learning_effects:", output)
                self.assertIn("behavior_weight_patch", output)
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
