from __future__ import annotations

import json
import time
import traceback
from typing import Any

from astrbot.api import logger


class ObservabilityLogger:
    def __init__(self, enabled: bool = True, max_field_chars: int = 2000):
        self.enabled = enabled
        self.max_field_chars = max_field_chars

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > self.max_field_chars:
                return value[: self.max_field_chars] + "...<truncated>"
            return value
        return repr(value)

    def _log(self, level: str, event: str, **kwargs: Any):
        if not self.enabled:
            return
        payload = {
            "event": event,
            "ts": int(time.time()),
            **{k: self._json_safe(v) for k, v in kwargs.items()},
        }
        text = json.dumps(payload, ensure_ascii=False)
        if level == "error":
            logger.error(text)
        else:
            logger.info(text)

    def info(self, event: str, **kwargs: Any):
        self._log("info", event, **kwargs)

    def error(self, event: str, **kwargs: Any):
        self._log("error", event, **kwargs)

    def exception(self, event: str, exc: BaseException, **kwargs: Any):
        self.error(
            event,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            **kwargs,
        )
