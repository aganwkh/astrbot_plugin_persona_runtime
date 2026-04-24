from __future__ import annotations

from collections import Counter, defaultdict

from .models import ExampleRecord, LearningBufferItem, TurnTraceRecord, WeightPatchRecord


class BatchLearningAnalyzer:
    def __init__(
        self,
        *,
        min_batch_size: int = 3,
        behavior_positive_step: float = 0.04,
        behavior_negative_step: float = 0.1,
        example_tag_positive_step: float = 0.05,
        example_tag_negative_step: float = 0.08,
    ):
        self.min_batch_size = min_batch_size
        self.behavior_positive_step = behavior_positive_step
        self.behavior_negative_step = behavior_negative_step
        self.example_tag_positive_step = example_tag_positive_step
        self.example_tag_negative_step = example_tag_negative_step

    def analyze(
        self,
        items: list[LearningBufferItem],
        traces_by_turn: dict[str, TurnTraceRecord],
        examples_by_id: dict[int, ExampleRecord],
        *,
        force: bool = False,
    ) -> list[WeightPatchRecord]:
        if not force and len(items) < self.min_batch_size:
            return []

        patches: list[WeightPatchRecord] = []
        items_by_scene: dict[str, list[LearningBufferItem]] = defaultdict(list)
        for item in items:
            items_by_scene[item.scene].append(item)

        for scene, scene_items in items_by_scene.items():
            if not force and len(scene_items) < 2:
                continue
            patches.extend(self._behavior_patches_for_scene(scene, scene_items))
            patches.extend(self._example_tag_patches_for_scene(scene, scene_items, traces_by_turn, examples_by_id))
        return patches

    def _behavior_patches_for_scene(self, scene: str, items: list[LearningBufferItem]) -> list[WeightPatchRecord]:
        patches: list[WeightPatchRecord] = []
        non_feedback = [item for item in items if not item.feedback_label]
        feedback = [item for item in items if item.feedback_label]

        if len(non_feedback) >= 2:
            behavior_counts = Counter(item.selected_behavior for item in non_feedback)
            dominant_behavior, dominant_count = behavior_counts.most_common(1)[0]
            if dominant_count / len(non_feedback) >= 0.65:
                patches.append(
                    WeightPatchRecord(
                        patch_type="behavior_weight_patch",
                        scene=scene,
                        target_key=dominant_behavior,
                        delta=self.behavior_positive_step,
                        min_value=0.05,
                        max_value=0.95,
                        evidence_count=dominant_count,
                        reason=f"reinforce dominant behavior from {dominant_count}/{len(non_feedback)} recent turns",
                        metadata={
                            "source": "batch_learning_analyzer",
                            "signal": "dominant_behavior",
                            "turn_ids": [item.turn_id for item in non_feedback],
                        },
                    )
                )

        feedback_counts = Counter(item.feedback_label for item in feedback if item.feedback_label)
        if feedback_counts.get("followup_question", 0) > 0:
            evidence = feedback_counts["followup_question"]
            patches.append(
                WeightPatchRecord(
                    patch_type="behavior_weight_patch",
                    scene=scene,
                    target_key="followup_question",
                    delta=-self.behavior_negative_step,
                    min_value=0.05,
                    max_value=0.95,
                    evidence_count=evidence,
                    reason=f"reduce follow-up pressure after {evidence} direct feedback turns",
                    metadata={
                        "source": "batch_learning_analyzer",
                        "signal": "followup_feedback",
                        "turn_ids": [item.turn_id for item in feedback if item.feedback_label == "followup_question"],
                    },
                )
            )
        if scene == "task_request" and feedback_counts.get("accuracy", 0) > 0:
            evidence = feedback_counts["accuracy"]
            patches.append(
                WeightPatchRecord(
                    patch_type="behavior_weight_patch",
                    scene=scene,
                    target_key="solution",
                    delta=self.behavior_positive_step,
                    min_value=0.05,
                    max_value=0.95,
                    evidence_count=evidence,
                    reason=f"reinforce direct solution behavior after {evidence} accuracy feedback turns",
                    metadata={
                        "source": "batch_learning_analyzer",
                        "signal": "accuracy_feedback",
                        "turn_ids": [item.turn_id for item in feedback if item.feedback_label == "accuracy"],
                    },
                )
            )
        return patches

    def _example_tag_patches_for_scene(
        self,
        scene: str,
        items: list[LearningBufferItem],
        traces_by_turn: dict[str, TurnTraceRecord],
        examples_by_id: dict[int, ExampleRecord],
    ) -> list[WeightPatchRecord]:
        positive_tag_counts: Counter[str] = Counter()
        negative_tag_counts: Counter[str] = Counter()
        positive_turns: dict[str, list[str]] = defaultdict(list)
        negative_turns: dict[str, list[str]] = defaultdict(list)

        for item in items:
            trace = traces_by_turn.get(item.turn_id)
            if trace is None:
                continue
            tags = self._selected_tags(trace, examples_by_id)
            if not tags:
                continue
            for tag in tags:
                if item.feedback_label:
                    negative_tag_counts[tag] += 1
                    negative_turns[tag].append(item.turn_id)
                else:
                    positive_tag_counts[tag] += 1
                    positive_turns[tag].append(item.turn_id)

        patches: list[WeightPatchRecord] = []
        for tag, count in positive_tag_counts.items():
            if count < 2:
                continue
            patches.append(
                WeightPatchRecord(
                    patch_type="example_tag_weight_patch",
                    scene=scene,
                    target_key=tag,
                    delta=self.example_tag_positive_step,
                    min_value=-0.4,
                    max_value=0.4,
                    evidence_count=count,
                    reason=f"boost example tag '{tag}' from {count} successful selections",
                    metadata={
                        "source": "batch_learning_analyzer",
                        "signal": "successful_example_tag",
                        "turn_ids": positive_turns[tag],
                    },
                )
            )
        for tag, count in negative_tag_counts.items():
            patches.append(
                WeightPatchRecord(
                    patch_type="example_tag_weight_patch",
                    scene=scene,
                    target_key=tag,
                    delta=-self.example_tag_negative_step,
                    min_value=-0.4,
                    max_value=0.4,
                    evidence_count=count,
                    reason=f"downweight example tag '{tag}' after {count} feedback-linked selections",
                    metadata={
                        "source": "batch_learning_analyzer",
                        "signal": "feedback_example_tag",
                        "turn_ids": negative_turns[tag],
                    },
                )
            )
        return patches

    @staticmethod
    def _selected_tags(trace: TurnTraceRecord, examples_by_id: dict[int, ExampleRecord]) -> list[str]:
        tags: list[str] = []
        for example_id in trace.selected_example_ids:
            example = examples_by_id.get(example_id)
            if example is None:
                continue
            tags.extend(tag.strip() for tag in example.tags if tag.strip())
        return tags
