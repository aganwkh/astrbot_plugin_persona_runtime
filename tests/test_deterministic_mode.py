from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_harness import FakeEvent, build_plugin
from persona_runtime.models import LearningBufferItem, TurnTraceRecord


class DeterministicModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_input_keeps_same_prompt_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, module = build_plugin(Path(tmp), timeout_seconds=1, extra_config={"deterministic_mode": True})
            try:
                event = FakeEvent(text="hello normal path", user_id="user-1", session_id="room-1")
                req_one = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req_one)
                turn_one = plugin.turn_registry.get(plugin.turn_registry.find_latest_turn_id(event))
                plan_one = turn_one.bundle.prompt_plan.to_log_dict()
                await plugin.on_llm_response(event, module.LLMResponse(text="first reply"))
                await plugin.after_message_sent(event)

                req_two = module.ProviderRequest(system_prompt="")
                await plugin.on_llm_request(event, req_two)
                turn_two = plugin.turn_registry.get(plugin.turn_registry.find_latest_turn_id(event))
                plan_two = turn_two.bundle.prompt_plan.to_log_dict()

                self.assertEqual(plan_one, plan_two)
            finally:
                await plugin.terminate()

    async def test_deterministic_mode_skips_background_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin, _module = build_plugin(Path(tmp), timeout_seconds=1, extra_config={"deterministic_mode": True})
            try:
                await plugin._ensure_ready()
                for index in range(3):
                    turn_id = f"turn-{index}"
                    await plugin.repository.create_turn_trace(
                        TurnTraceRecord(
                            turn_id=turn_id,
                            scene="casual_chat",
                            selected_behavior="followup_question",
                            behavior_probabilities={"followup_question": 0.58},
                            scene_scores={"casual_chat": 0.4},
                            selected_example_ids=[],
                            selected_lore_ids=[],
                        )
                    )
                    await plugin.repository.create_learning_buffer_item(
                        LearningBufferItem(
                            turn_id=turn_id,
                            user_excerpt="hello",
                            assistant_excerpt="hi",
                            scene="casual_chat",
                            selected_behavior="followup_question",
                        )
                    )

                summary = await plugin._maybe_run_learning_analysis()
                patches = await plugin.repository.list_weight_patches(limit=10)
                pending = await plugin.repository.list_pending_learning_buffer(limit=10)

                self.assertEqual(summary["created_patches"], [])
                self.assertEqual(patches, [])
                self.assertEqual(len(pending), 3)
            finally:
                await plugin.terminate()


if __name__ == "__main__":
    unittest.main()
