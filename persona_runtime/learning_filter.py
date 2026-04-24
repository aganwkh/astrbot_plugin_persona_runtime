from __future__ import annotations

import re

from .models import LearningBufferItem, RuntimeContextBundle


class LearningFilter:
    def __init__(
        self,
        *,
        max_user_chars: int = 500,
        max_assistant_chars: int = 800,
        max_total_turn_chars: int = 1200,
        skip_if_code_block_chars_over: int = 300,
        skip_if_html_chars_over: int = 300,
        skip_if_log_chars_over: int = 500,
        skip_if_json_chars_over: int = 500,
        allow_feedback_even_if_long: bool = True,
        keep_feedback_excerpt_chars: int = 200,
    ):
        self.max_user_chars = max_user_chars
        self.max_assistant_chars = max_assistant_chars
        self.max_total_turn_chars = max_total_turn_chars
        self.skip_if_code_block_chars_over = skip_if_code_block_chars_over
        self.skip_if_html_chars_over = skip_if_html_chars_over
        self.skip_if_log_chars_over = skip_if_log_chars_over
        self.skip_if_json_chars_over = skip_if_json_chars_over
        self.allow_feedback_even_if_long = allow_feedback_even_if_long
        self.keep_feedback_excerpt_chars = keep_feedback_excerpt_chars

    def build_item(self, bundle: RuntimeContextBundle) -> LearningBufferItem | None:
        user_text = (bundle.normalized_input.raw_text or "").strip()
        assistant_text = (bundle.assistant_reply_text or "").strip()
        if not user_text or not assistant_text:
            return None

        too_long = (
            len(user_text) > self.max_user_chars
            or len(assistant_text) > self.max_assistant_chars
            or (len(user_text) + len(assistant_text)) > self.max_total_turn_chars
        )
        code_like = self._code_like_chars(user_text, assistant_text) > self.skip_if_code_block_chars_over
        html_like = self._html_like_chars(user_text, assistant_text) > self.skip_if_html_chars_over
        log_like = self._log_like_chars(user_text, assistant_text) > self.skip_if_log_chars_over
        json_like = self._json_like_chars(user_text, assistant_text) > self.skip_if_json_chars_over

        if not too_long and not any([code_like, html_like, log_like, json_like]):
            return LearningBufferItem(
                turn_id=bundle.turn_id,
                user_excerpt=user_text,
                assistant_excerpt=assistant_text,
                scene=bundle.prompt_plan.main_scene,
                selected_behavior=bundle.prompt_plan.selected_behavior,
                feedback_label=bundle.scene_resolution.feedback_target if bundle.scene_resolution.is_feedback else None,
                content_mode="full_short",
                learning_eligible=True,
            )

        if self.allow_feedback_even_if_long and bundle.scene_resolution.is_feedback:
            return LearningBufferItem(
                turn_id=bundle.turn_id,
                user_excerpt=self._truncate(user_text, self.keep_feedback_excerpt_chars),
                assistant_excerpt=self._truncate(assistant_text, self.keep_feedback_excerpt_chars),
                scene=bundle.prompt_plan.main_scene,
                selected_behavior=bundle.prompt_plan.selected_behavior,
                feedback_label=bundle.scene_resolution.feedback_target,
                content_mode="feedback_excerpt",
                learning_eligible=True,
            )
        return None

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        suffix = "...<trimmed>"
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)].rstrip() + suffix

    @staticmethod
    def _code_like_chars(*texts: str) -> int:
        total = 0
        for text in texts:
            if "```" in text:
                total += len(text)
            total += sum(len(line) for line in text.splitlines() if re.match(r"^\s*(def |class |if |for |while |return |import |\{|\}|\[|\])", line))
        return total

    @staticmethod
    def _html_like_chars(*texts: str) -> int:
        total = 0
        for text in texts:
            if re.search(r"</?[a-zA-Z][^>]{0,40}>", text):
                total += len(text)
        return total

    @staticmethod
    def _log_like_chars(*texts: str) -> int:
        total = 0
        for text in texts:
            total += sum(
                len(line)
                for line in text.splitlines()
                if re.search(r"\b(INFO|WARN|ERROR|DEBUG|TRACE)\b", line) or re.search(r"\d{2}:\d{2}:\d{2}", line)
            )
        return total

    @staticmethod
    def _json_like_chars(*texts: str) -> int:
        total = 0
        for text in texts:
            compact = text.strip()
            if compact.startswith("{") and compact.endswith("}"):
                total += len(compact)
            if compact.startswith("[") and compact.endswith("]"):
                total += len(compact)
        return total
