from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = BASE_DIR.name


def install_astrbot_mock():
    if "astrbot.api" in sys.modules:
        return

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    provider_mod = types.ModuleType("astrbot.api.provider")
    star_mod = types.ModuleType("astrbot.api.star")

    class MockLogger:
        def info(self, text: str):
            print(text)

        def error(self, text: str):
            print(text, file=sys.stderr)

    class AstrBotConfig(dict):
        pass

    class AstrMessageEvent:
        pass

    class ProviderRequest:
        def __init__(self, system_prompt: str = ""):
            self.system_prompt = system_prompt

    class LLMResponse:
        pass

    class Context:
        pass

    class Star:
        def __init__(self, context: Context):
            self.context = context

    class Filter:
        @staticmethod
        def _identity_decorator(*_args: Any, **_kwargs: Any):
            def decorate(func):
                return func

            return decorate

        command = _identity_decorator
        on_llm_request = _identity_decorator
        on_agent_begin = _identity_decorator
        on_using_llm_tool = _identity_decorator
        on_llm_tool_respond = _identity_decorator
        on_llm_response = _identity_decorator
        on_agent_done = _identity_decorator
        after_message_sent = _identity_decorator

    api_mod.AstrBotConfig = AstrBotConfig
    api_mod.logger = MockLogger()
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.filter = Filter()
    provider_mod.ProviderRequest = ProviderRequest
    provider_mod.LLMResponse = LLMResponse
    star_mod.Context = Context
    star_mod.Star = Star

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.provider"] = provider_mod
    sys.modules["astrbot.api.star"] = star_mod


class FakeMessageObj:
    def __init__(self, platform_name: str, session_id: str):
        self.platform_name = platform_name
        self.session_id = session_id


class FakeEvent:
    def __init__(
        self,
        *,
        text: str,
        user_id: str,
        session_id: str,
        platform_name: str = "dev",
        streaming: bool = False,
    ):
        self.message_str = text
        self.sender_id = user_id
        self.session_id = session_id
        self.platform_name = platform_name
        self.unified_msg_origin = f"{platform_name}:{session_id}:{user_id}"
        self.message_obj = FakeMessageObj(platform_name, session_id)
        self.streaming = streaming

    def get_sender_id(self) -> str:
        return self.sender_id

    def plain_result(self, text: str) -> str:
        return text


class FakeTool:
    def __init__(self, name: str):
        self.name = name


class FakeToolResult:
    def __init__(self, content: list[str] | None):
        self.content = content


def import_plugin_main():
    install_astrbot_mock()
    parent = str(BASE_DIR.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = __import__(f"{PACKAGE_NAME}.main", fromlist=["Main"])
    return module


def build_plugin(work_dir: Path, timeout_seconds: int):
    module = import_plugin_main()
    config = module.AstrBotConfig(
        {
            "enable_runtime": True,
            "DEBUG_MODE": True,
            "commit_timeout_seconds": timeout_seconds,
            "max_runtime_chars": 800,
            "max_lore_chars": 300,
            "lore_top_k": 2,
            "lore_score_threshold": 0.55,
            "static_persona_prompt": "",
        }
    )
    plugin = module.Main(module.Context(), config)

    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    source_lore = BASE_DIR / "data" / "lore_keywords.json"
    if source_lore.exists():
        shutil.copyfile(source_lore, data_dir / "lore_keywords.json")

    plugin.data_dir = data_dir
    plugin.lore_injector = module.LoreInjector(data_dir / "lore_keywords.json")
    plugin.repository = module.StateRepository(data_dir / "persona_runtime.db")
    plugin.commit_watchdog = module.CommitWatchdog(
        turn_registry=plugin.turn_registry,
        timeout_seconds=timeout_seconds,
        on_timeout=plugin._rollback_turn,
        logger=plugin.logger,
    )
    return plugin, module


async def normal_turn(plugin: Any, module: Any, event: FakeEvent):
    req = module.ProviderRequest(system_prompt="base system prompt")
    await plugin.on_llm_request(event, req)
    await plugin.on_llm_response(event, module.LLMResponse())
    await plugin.after_message_sent(event)
    print(f"[normal] injected_system_prompt_len={len(req.system_prompt)}")


async def tool_turn(plugin: Any, module: Any, event: FakeEvent):
    req = module.ProviderRequest(system_prompt="")
    tool = FakeTool("mock_lookup")
    await plugin.on_llm_request(event, req)
    await plugin.on_agent_begin(event, run_context={})
    await plugin.on_using_llm_tool(event, tool, {"query": "dev harness"})
    await plugin.on_llm_tool_respond(event, tool, {"query": "dev harness"}, FakeToolResult(["ok"]))
    await plugin.on_agent_done(event, run_context={}, resp=module.LLMResponse())
    await plugin.after_message_sent(event)
    print(f"[tool] injected_system_prompt_len={len(req.system_prompt)}")


async def timeout_turn(plugin: Any, module: Any, event: FakeEvent, timeout_seconds: int):
    req = module.ProviderRequest(system_prompt="")
    await plugin.on_llm_request(event, req)
    await plugin.on_llm_response(event, module.LLMResponse())
    await asyncio.sleep(timeout_seconds + 1.5)
    print("[timeout] after_message_sent intentionally skipped")


async def concurrent_turns(plugin: Any, module: Any):
    same_session = [
        FakeEvent(text="same session turn A", user_id="user-a", session_id="room-1"),
        FakeEvent(text="same session turn B", user_id="user-b", session_id="room-1"),
    ]
    cross_session = [
        FakeEvent(text="cross session turn A", user_id="user-c", session_id="room-2"),
        FakeEvent(text="cross session turn B", user_id="user-c", session_id="room-3"),
    ]
    await asyncio.gather(*(normal_turn(plugin, module, event) for event in same_session))
    await asyncio.gather(*(normal_turn(plugin, module, event) for event in cross_session))


async def run_scenario(name: str, timeout_seconds: int):
    with tempfile.TemporaryDirectory(prefix="persona_runtime_harness_") as tmp:
        plugin, module = build_plugin(Path(tmp), timeout_seconds)
        try:
            if name in {"normal", "all"}:
                await normal_turn(
                    plugin,
                    module,
                    FakeEvent(text="hello normal path", user_id="user-1", session_id="room-normal"),
                )
            if name in {"tool", "all"}:
                await tool_turn(
                    plugin,
                    module,
                    FakeEvent(text="please use a tool", user_id="user-2", session_id="room-tool"),
                )
            if name in {"timeout", "all"}:
                await timeout_turn(
                    plugin,
                    module,
                    FakeEvent(text="timeout path", user_id="user-3", session_id="room-timeout"),
                    timeout_seconds,
                )
            if name in {"concurrent", "all"}:
                await concurrent_turns(plugin, module)
            print("[snapshot]")
            print(plugin.turn_registry.snapshot())
        finally:
            await plugin.terminate()


def main():
    parser = argparse.ArgumentParser(description="Offline lifecycle harness for astrbot_plugin_persona_runtime.")
    parser.add_argument(
        "--scenario",
        choices=["normal", "tool", "timeout", "concurrent", "all"],
        default="all",
        help="Lifecycle scenario to run.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1, help="Watchdog timeout for harness runs.")
    args = parser.parse_args()
    asyncio.run(run_scenario(args.scenario, args.timeout_seconds))


if __name__ == "__main__":
    main()
