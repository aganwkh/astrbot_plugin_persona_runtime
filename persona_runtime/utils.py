from __future__ import annotations

import hashlib
from typing import Any


def _stable(obj: Any, default: str = "unknown") -> str:
    if obj is None:
        return default
    try:
        text = str(obj).strip()
        return text or default
    except Exception:  # noqa: BLE001
        return default


def build_scope_key(*parts: Any) -> str:
    raw = "|".join(_stable(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{raw}|{digest}"


def safe_get_sender_id(event: Any) -> str:
    for name in ["get_sender_id", "sender_id"]:
        obj = getattr(event, name, None)
        if callable(obj):
            try:
                return _stable(obj())
            except Exception:  # noqa: BLE001
                continue
        if obj is not None:
            return _stable(obj)
    return "unknown_user"


def safe_get_sender_name(event: Any) -> str:
    for name in ["get_sender_name", "sender_name"]:
        obj = getattr(event, name, None)
        if callable(obj):
            try:
                return _stable(obj())
            except Exception:  # noqa: BLE001
                continue
        if obj is not None:
            return _stable(obj)
    return "unknown_sender"


def safe_get_unified_msg_origin(event: Any) -> str:
    value = getattr(event, "unified_msg_origin", None)
    return _stable(value, "unknown_origin")


def safe_get_platform_id(event: Any) -> str:
    for name in ["platform_name", "adapter_name"]:
        value = getattr(event, name, None)
        if value is not None:
            return _stable(value)
    msg_obj = getattr(event, "message_obj", None)
    for name in ["platform_name", "adapter_name", "adapter_type"]:
        value = getattr(msg_obj, name, None)
        if value is not None:
            return _stable(value)
    return "unknown_platform"


def safe_get_conversation_id(event: Any) -> str:
    msg_obj = getattr(event, "message_obj", None)
    for candidate in [
        getattr(msg_obj, "session_id", None),
        getattr(msg_obj, "group_id", None),
        getattr(msg_obj, "conversation_id", None),
        getattr(event, "session_id", None),
        safe_get_unified_msg_origin(event),
    ]:
        if candidate is not None:
            text = _stable(candidate)
            if text and text != "unknown_origin":
                return text
    return safe_get_unified_msg_origin(event)
