from __future__ import annotations

from .models import DialoguePolicyResult, LoreItem, MergedPromptPayload


class PromptMergePolicy:
    def merge(self, policy: DialoguePolicyResult, lore_items: list[LoreItem]) -> MergedPromptPayload:
        lines: list[str] = []
        style_lines: list[str] = []

        if policy.reply_mode == "short":
            lines.append("Prefer a concise answer that addresses the user's immediate context first.")
        elif policy.reply_mode == "task":
            lines.append("Use task mode: give the conclusion first, then only the necessary details.")
        else:
            lines.append("Respond to the user's immediate context, then answer the core request.")

        if not policy.allow_followup_question:
            lines.append("Do not ask follow-up questions unless the task is blocked.")
        if policy.need_soften_tool_tone:
            lines.append("If tools or lookups are mentioned, keep the wording natural and avoid system-report style.")
        if policy.tone_mode == "bright":
            style_lines.append("Tone may be lightly upbeat, without becoming exaggerated.")
        elif policy.tone_mode == "defensive":
            style_lines.append("Tone may be firm and guarded, without losing control.")

        return MergedPromptPayload(policy_lines=lines, lore_items=lore_items, style_lines=style_lines)
