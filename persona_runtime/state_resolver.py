from __future__ import annotations

from .models import NormalizedInput, SceneResolution, SessionState, UserState


class StateResolver:
    SCENES = ("status_share", "casual_chat", "complaint", "task_request")
    STATUS_HINTS = {
        "刚吃完饭",
        "刚下课",
        "到家了",
        "回来了",
        "睡觉",
        "在忙",
        "有点累",
        "刚醒",
        "鍒氬悆瀹岄キ",
        "鍒氫笅璇",
        "鍒板",
        "鍥炴潵",
        "鐫¤",
        "鍦ㄥ繖",
        "鏈夌偣绱",
        "鍒氶啋",
    }
    TASK_HINTS = {
        "帮我",
        "你看看",
        "看看",
        "查一下",
        "分析",
        "整理",
        "修",
        "改",
        "代码",
        "直接改",
        "甯垜",
        "浣犵湅鐪",
        "鐪嬬湅",
        "鏌ヤ竴涓",
        "鍒嗘瀽",
        "鏁寸悊",
        "淇",
        "鏀",
        "浠ｇ爜",
        "鐩存帴鏀",
    }
    COMPLAINT_HINTS = {
        "我服了",
        "气死了",
        "崩溃",
        "离谱",
        "无语",
        "难受",
        "算了",
        "不对",
        "鎴戞湇浜",
        "鐑",
        "宕╂簝",
        "绂昏氨",
        "鏃犺",
        "闅惧彈",
        "绠椾簡",
        "涓嶅",
    }
    DOUBT_HINTS = {
        "你刚刚是不是乱说",
        "乱说",
        "感觉不对",
        "不靠谱",
        "别一直问我",
        "别问了",
        "鏄笉鏄贡璇",
        "鎰熻涓嶅",
        "涓嶉潬璋",
        "鍒竴鐩撮棶鎴",
        "鍒棶浜",
    }

    def resolve(self, normalized: NormalizedInput, user_state: UserState, session_state: SessionState) -> SceneResolution:
        text = normalized.raw_text.strip()
        scene_scores = {
            "status_share": self._status_share_score(text),
            "casual_chat": self._casual_chat_score(text),
            "complaint": self._complaint_score(text),
            "task_request": self._task_request_score(text),
        }
        main_scene = max(self.SCENES, key=lambda scene: (scene_scores[scene], scene))
        is_feedback, feedback_target = self._feedback_signal(text)
        feature_scores = {
            "status_share_score": scene_scores["status_share"],
            "casual_chat_score": scene_scores["casual_chat"],
            "complaint_score": scene_scores["complaint"],
            "task_score": scene_scores["task_request"],
            "is_feedback_score": 1.0 if is_feedback else 0.0,
            "tool_style_bias": min(1.0, user_state.tool_style_bias / 10.0),
            "session_turn_counter": float(min(session_state.turn_counter, 20)),
        }
        reasons = {
            "status_share": self._reason_for_status_share(text),
            "casual_chat": self._reason_for_casual_chat(text),
            "complaint": self._reason_for_complaint(text),
            "task_request": self._reason_for_task_request(text),
        }
        return SceneResolution(
            main_scene=main_scene,
            scene_scores=scene_scores,
            is_feedback=is_feedback,
            feedback_target=feedback_target,
            feature_scores=feature_scores,
            reasons=reasons,
        )

    def _status_share_score(self, text: str) -> float:
        score = 0.18
        if self._contains_any(text, {"刚", "刚刚", "刚才", "鍒", "鍒氬垰", "鍒氭墠"}):
            score += 0.22
        if self._contains_any(text, self.STATUS_HINTS):
            score += 0.48
        if len(text) <= 10:
            score += 0.12
        if self._has_question(text):
            score -= 0.12
        if self._contains_any(text, self.DOUBT_HINTS):
            score -= 0.28
        if self._contains_any(text, self.TASK_HINTS):
            score -= 0.2
        return self._clamp(score)

    def _casual_chat_score(self, text: str) -> float:
        score = 0.4
        if len(text) <= 6:
            score += 0.08
        if self._has_question(text):
            score += 0.1
        if self._contains_any(text, self.TASK_HINTS):
            score -= 0.18
        if self._contains_any(text, self.COMPLAINT_HINTS.union(self.DOUBT_HINTS)):
            score -= 0.12
        return self._clamp(score)

    def _complaint_score(self, text: str) -> float:
        score = 0.08
        if self._contains_any(text, self.COMPLAINT_HINTS):
            score += 0.6
        if self._contains_any(text, self.DOUBT_HINTS):
            score += 0.42
        if self._has_exclamation(text):
            score += 0.05
        if self._contains_any(text, self.TASK_HINTS):
            score -= 0.08
        return self._clamp(score)

    def _task_request_score(self, text: str) -> float:
        score = 0.05
        if self._contains_any(text, self.TASK_HINTS):
            score += 0.7
        if self._contains_any(text, {"怎么", "如何", "方案", "总结", "解释", "鎬庝箞", "濡備綍", "鏂规", "鎬荤粨", "瑙ｉ噴"}):
            score += 0.15
        if self._contains_any(text, self.STATUS_HINTS):
            score -= 0.18
        return self._clamp(score)

    def _feedback_signal(self, text: str) -> tuple[bool, str]:
        if self._contains_any(text, {"别一直问我", "别问了", "不要一直问", "鍒竴鐩撮棶鎴", "鍒棶浜", "涓嶈涓€鐩撮棶"}):
            return True, "followup_question"
        if self._contains_any(text, {"这版好多了", "这样就对了", "好多了", "杩欑増濂藉浜", "杩欐牱灏卞浜", "濂藉浜"}):
            return True, "overall_reply"
        if self._contains_any(text, self.DOUBT_HINTS.union({"不对", "涓嶅"})):
            return True, "accuracy"
        return False, "none"

    def _reason_for_status_share(self, text: str) -> str:
        if self._contains_any(text, self.DOUBT_HINTS):
            return "status-share prior reduced because the utterance questions the previous reply"
        if self._contains_any(text, self.STATUS_HINTS):
            return "detected recent-life update keywords"
        if self._contains_any(text, {"刚", "刚刚", "刚才", "鍒", "鍒氬垰"}):
            return "detected immediate status wording"
        return "baseline short-message status-share prior"

    @staticmethod
    def _reason_for_casual_chat(text: str) -> str:
        if "?" in text or "？" in text:
            return "detected conversational question"
        return "default fallback scene for ordinary short chat"

    def _reason_for_complaint(self, text: str) -> str:
        if self._contains_any(text, self.DOUBT_HINTS):
            return "detected direct doubt about the previous reply accuracy"
        if self._contains_any(text, self.COMPLAINT_HINTS):
            return "detected complaint or frustration wording"
        return "low complaint prior"

    def _reason_for_task_request(self, text: str) -> str:
        if self._contains_any(text, self.TASK_HINTS):
            return "detected explicit request-for-help wording"
        return "low task prior"

    @staticmethod
    def _contains_any(text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords if keyword)

    @staticmethod
    def _has_question(text: str) -> bool:
        return "?" in text or "？" in text or "锛" in text

    @staticmethod
    def _has_exclamation(text: str) -> bool:
        return "!" in text or "！" in text or "锛" in text

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.01, min(round(value, 3), 0.99))
