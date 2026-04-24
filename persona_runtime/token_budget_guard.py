from __future__ import annotations

import copy

from .models import PromptPlan


class TokenBudgetGuard:
    def __init__(self, max_runtime_chars: int = 800):
        self.max_runtime_chars = max_runtime_chars

    def apply(self, plan: PromptPlan) -> PromptPlan:
        budgeted = copy.deepcopy(plan)
        if self.max_runtime_chars <= 0:
            budgeted.current_scene_text = ""
            budgeted.behavior_tendency_text = ""
            budgeted.selected_modules = []
            budgeted.selected_lore_ids = []
            budgeted.token_budget.max_runtime_chars = self.max_runtime_chars
            budgeted.token_budget.used_chars = 0
            budgeted.token_budget.trimmed = True
            return budgeted

        remaining = self.max_runtime_chars
        trimmed = False

        module_order = ["base_persona", "base_style"]
        module_map = {module.module_id: module for module in budgeted.selected_modules}

        for module_id in module_order:
            module = module_map.get(module_id)
            if module is None:
                continue
            kept_text, was_trimmed = self._fit_text(module.content, remaining)
            trimmed = trimmed or was_trimmed
            module.content = kept_text
            remaining -= len(kept_text)

        budgeted.current_scene_text, current_trimmed = self._fit_text(budgeted.current_scene_text, remaining)
        trimmed = trimmed or current_trimmed
        remaining -= len(budgeted.current_scene_text)

        budgeted.behavior_tendency_text, behavior_trimmed = self._fit_text(budgeted.behavior_tendency_text, remaining)
        trimmed = trimmed or behavior_trimmed
        remaining -= len(budgeted.behavior_tendency_text)

        kept_examples = []
        for example in budgeted.selected_examples:
            example_text = f"用户：{example.user_message}\n角色：{example.assistant_reply}"
            kept_text, was_trimmed = self._fit_text(example_text, remaining)
            if not kept_text or was_trimmed:
                trimmed = trimmed or bool(example_text)
                break
            kept_examples.append(example)
            remaining -= len(kept_text)
        if len(kept_examples) != len(budgeted.selected_examples):
            trimmed = True
        budgeted.selected_examples = kept_examples

        for module_id in ["lore", "negative_pattern_reminder"]:
            module = module_map.get(module_id)
            if module is None:
                continue
            kept_text, was_trimmed = self._fit_text(module.content, remaining)
            trimmed = trimmed or was_trimmed
            module.content = kept_text
            remaining -= len(kept_text)

        filtered_modules = [module for module in budgeted.selected_modules if module.content]
        lore_kept = any(module.module_id == "lore" for module in filtered_modules)
        if not lore_kept:
            budgeted.selected_lore_ids = []
        budgeted.selected_modules = filtered_modules

        used_chars = self.max_runtime_chars - remaining
        budgeted.token_budget.max_runtime_chars = self.max_runtime_chars
        budgeted.token_budget.used_chars = max(0, used_chars)
        budgeted.token_budget.trimmed = trimmed
        return budgeted

    @staticmethod
    def _fit_text(text: str, remaining: int) -> tuple[str, bool]:
        if not text or remaining <= 0:
            return "", bool(text)
        if len(text) <= remaining:
            return text, False
        suffix = "...<trimmed>"
        if remaining <= len(suffix):
            return text[:remaining], True
        return text[: remaining - len(suffix)].rstrip() + suffix, True
