from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .models import PromptPlan


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    user_text: str
    expected_scene: str
    expected_behavior: str
    forbidden_behaviors: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    case: EvaluationCase
    prompt_plan: PromptPlan
    passed: bool
    reasons: list[str]


class EvaluationSuite:
    def __init__(self):
        self.cases = [
            EvaluationCase(
                case_id="status_share_meal",
                user_text="刚吃完饭",
                expected_scene="status_share",
                expected_behavior="short_ack",
                forbidden_behaviors=["followup_question"],
            ),
            EvaluationCase(
                case_id="complaint_frustration",
                user_text="我服了",
                expected_scene="complaint",
                expected_behavior="comfort",
                forbidden_behaviors=["followup_question"],
            ),
            EvaluationCase(
                case_id="task_request_code",
                user_text="帮我看看这个代码",
                expected_scene="task_request",
                expected_behavior="solution",
            ),
            EvaluationCase(
                case_id="casual_doubt",
                user_text="你刚刚是不是乱说",
                expected_scene="complaint",
                expected_behavior="comfort",
                forbidden_behaviors=["followup_question"],
            ),
        ]

    async def run(
        self,
        build_plan: Callable[[str], "PromptPlan | None"],
    ) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for case in self.cases:
            plan = await build_plan(case.user_text)
            reasons: list[str] = []
            passed = True
            if plan is None:
                passed = False
                reasons.append("plan was not generated")
                results.append(
                    EvaluationResult(
                        case=case,
                        prompt_plan=PromptPlan(main_scene="none", scene_scores={}),
                        passed=False,
                        reasons=reasons,
                    )
                )
                continue
            if plan.main_scene != case.expected_scene:
                passed = False
                reasons.append(f"expected scene {case.expected_scene}, got {plan.main_scene}")
            if plan.selected_behavior != case.expected_behavior:
                passed = False
                reasons.append(f"expected behavior {case.expected_behavior}, got {plan.selected_behavior}")
            for forbidden in case.forbidden_behaviors:
                if plan.selected_behavior == forbidden:
                    passed = False
                    reasons.append(f"forbidden behavior selected: {forbidden}")
            results.append(EvaluationResult(case=case, prompt_plan=plan, passed=passed, reasons=reasons))
        return results
