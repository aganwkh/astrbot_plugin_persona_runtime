from __future__ import annotations

from .models import ExampleRecord, ExampleSelection, LearningEffect, NormalizedInput, PromptPlan, WeightPatchRecord


class ExampleSelector:
    def __init__(
        self,
        min_examples_per_turn: int = 0,
        max_examples_per_turn: int = 3,
        max_example_chars_total: int = 600,
        max_single_example_chars: int = 220,
        min_score: float = 0.35,
    ):
        self.min_examples_per_turn = min_examples_per_turn
        self.max_examples_per_turn = max_examples_per_turn
        self.max_example_chars_total = max_example_chars_total
        self.max_single_example_chars = max_single_example_chars
        self.min_score = min_score

    def select(
        self,
        normalized: NormalizedInput,
        prompt_plan: PromptPlan,
        examples: list[ExampleRecord],
        patches: list[WeightPatchRecord] | None = None,
    ) -> list[ExampleSelection]:
        if not examples:
            return []

        query_terms = self._terms(normalized.raw_text)
        selected: list[ExampleSelection] = []
        used_chars = 0

        ranked = sorted(
            (
                self._score_example(example, prompt_plan.main_scene, query_terms)
                if patches is None
                else self._score_example(example, prompt_plan.main_scene, query_terms, patches)
                for example in examples
                if example.enabled
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        for score, reason, effects, example in ranked:
            if len(selected) >= self.max_examples_per_turn:
                break
            single_chars = len(example.user_message) + len(example.assistant_reply)
            if single_chars > self.max_single_example_chars:
                continue
            if used_chars + single_chars > self.max_example_chars_total:
                continue
            if score < self.min_score and len(selected) >= self.min_examples_per_turn:
                continue
            selected.append(
                ExampleSelection(
                    example_id=int(example.example_id or 0),
                    scene=example.scene,
                    tags=example.tags,
                    user_message=example.user_message,
                    assistant_reply=example.assistant_reply,
                    score=round(score, 3),
                    reason=reason,
                    applied_effects=effects,
                )
            )
            used_chars += single_chars
        return selected

    def _score_example(
        self,
        example: ExampleRecord,
        scene: str,
        query_terms: set[str],
        patches: list[WeightPatchRecord] | None = None,
    ) -> tuple[float, str, list[LearningEffect], ExampleRecord]:
        scene_match = 1.0 if example.scene == scene else 0.2
        tag_match = self._tag_match(example.tags, scene)
        text_similarity = self._similarity(query_terms, self._terms(example.user_message))
        quality = min(1.0, max(0.0, example.quality_score))
        patch_bonus, effects = self._tag_patch_bonus(example, scene, patches or [])
        score = (scene_match * 0.4) + (tag_match * 0.25) + (text_similarity * 0.2) + (quality * 0.1) + patch_bonus
        reason = (
            f"scene={example.scene} match={scene_match:.2f}; "
            f"tags={','.join(example.tags) or 'none'} tag_match={tag_match:.2f}; "
            f"text_similarity={text_similarity:.2f}"
        )
        if patch_bonus:
            reason += f"; learning_bonus={patch_bonus:.2f}"
        return score, reason, effects, example

    @staticmethod
    def _tag_patch_bonus(
        example: ExampleRecord,
        scene: str,
        patches: list[WeightPatchRecord],
    ) -> tuple[float, list[LearningEffect]]:
        if not example.tags or not patches:
            return 0.0, []
        tags_lower = {tag.strip().lower(): tag.strip() for tag in example.tags if tag.strip()}
        bonus = 0.0
        effects: list[LearningEffect] = []
        for patch in patches:
            if patch.patch_type != "example_tag_weight_patch" or patch.scene != scene:
                continue
            normalized_target = patch.target_key.strip().lower()
            if normalized_target not in tags_lower:
                continue
            effective_delta = round(patch.delta * patch.decay_factor, 3)
            bonus += effective_delta
            bonus = max(patch.min_value, min(bonus, patch.max_value))
            effects.append(
                LearningEffect(
                    patch_id=patch.patch_id,
                    patch_type=patch.patch_type,
                    scene=patch.scene,
                    target_key=patch.target_key,
                    delta=effective_delta,
                    reason=patch.reason,
                )
            )
        return bonus, effects

    @staticmethod
    def _tag_match(tags: list[str], scene: str) -> float:
        tags_lower = {tag.strip().lower() for tag in tags if tag.strip()}
        preferred = {
            "status_share": {"短句", "不追问", "克制"},
            "complaint": {"低压回应", "安抚", "克制"},
            "task_request": {"直接", "结论先行", "办事"},
            "casual_chat": {"自然", "连续对话", "轻松"},
        }.get(scene, set())
        if not tags_lower or not preferred:
            return 0.2
        hits = len(tags_lower.intersection({item.lower() for item in preferred}))
        if hits == 0:
            return 0.2
        return min(1.0, 0.35 + (hits * 0.25))

    @staticmethod
    def _terms(text: str) -> set[str]:
        clean = (text or "").replace("，", " ").replace("。", " ").replace("？", " ").replace("！", " ").replace(",", " ")
        chunks = [chunk.strip().lower() for chunk in clean.split() if chunk.strip()]
        if chunks:
            return set(chunks)
        # Chinese text often arrives without spaces; fall back to 2-char shingles.
        compact = clean.replace(" ", "")
        if len(compact) <= 2:
            return {compact} if compact else set()
        return {compact[index : index + 2] for index in range(len(compact) - 1)}

    @staticmethod
    def _similarity(query_terms: set[str], example_terms: set[str]) -> float:
        if not query_terms or not example_terms:
            return 0.0
        overlap = len(query_terms.intersection(example_terms))
        union = len(query_terms.union(example_terms))
        if union == 0:
            return 0.0
        return min(1.0, overlap / union)
