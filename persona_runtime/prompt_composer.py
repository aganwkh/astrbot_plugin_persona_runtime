from __future__ import annotations

from .models import PromptPlan


class PromptComposer:
    def compose(self, plan: PromptPlan) -> str:
        module_map = {module.module_id: module for module in plan.selected_modules}
        parts: list[str] = []

        base_persona = module_map.get("base_persona")
        if base_persona and base_persona.content:
            parts.append("[Core Persona]")
            parts.append(base_persona.content)

        base_style = module_map.get("base_style")
        if base_style and base_style.content:
            parts.append("[Base Style]")
            parts.append(base_style.content)

        if plan.current_scene_text:
            parts.append("[Current Scene]")
            parts.append(plan.current_scene_text)

        if plan.behavior_tendency_text:
            parts.append("[Behavior Tendency]")
            parts.append(plan.behavior_tendency_text)

        if plan.selected_examples:
            parts.append("[Relevant Examples]")
            for example in plan.selected_examples:
                parts.append(f"用户：{example.user_message}")
                parts.append(f"角色：{example.assistant_reply}")

        lore = module_map.get("lore")
        if lore and lore.content:
            parts.append("[Relevant Lore]")
            parts.append(lore.content)

        negative = module_map.get("negative_pattern_reminder")
        if negative and negative.content:
            parts.append("[Negative Pattern Reminder]")
            parts.append(negative.content)

        return "\n".join(parts).strip()
