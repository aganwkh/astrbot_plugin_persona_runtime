from __future__ import annotations

import unittest

from persona_runtime.models import NormalizedInput, SessionState, UserState
from persona_runtime.state_resolver import StateResolver


class StateResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolver = StateResolver()
        self.user_state = UserState(scope_key="user:1")
        self.session_state = SessionState(scope_key="session:1")

    def test_detects_status_share(self):
        result = self.resolver.resolve(NormalizedInput(raw_text="刚吃完饭"), self.user_state, self.session_state)
        self.assertEqual(result.main_scene, "status_share")
        self.assertGreater(result.scene_scores["status_share"], result.scene_scores["casual_chat"])

    def test_detects_complaint(self):
        result = self.resolver.resolve(NormalizedInput(raw_text="我服了"), self.user_state, self.session_state)
        self.assertEqual(result.main_scene, "complaint")
        self.assertGreater(result.scene_scores["complaint"], result.scene_scores["casual_chat"])

    def test_detects_task_request(self):
        result = self.resolver.resolve(NormalizedInput(raw_text="帮我看看这个代码"), self.user_state, self.session_state)
        self.assertEqual(result.main_scene, "task_request")
        self.assertGreater(result.scene_scores["task_request"], result.scene_scores["status_share"])

    def test_falls_back_to_casual_chat(self):
        result = self.resolver.resolve(NormalizedInput(raw_text="今天过得怎么样？"), self.user_state, self.session_state)
        self.assertEqual(result.main_scene, "casual_chat")
        self.assertFalse(result.is_feedback)


if __name__ == "__main__":
    unittest.main()
