# astrbot_plugin_persona_runtime (V3.2 starter)

这是一个和 `PROJECT_START_V3.2` 对齐的 AstrBot 插件 starter。

它仍然是 **starter，不是最终成品**。当前重点是把人格运行层的工程骨架、SQLite 状态仓储、两阶段提交、turn 超时回滚、工具态关联、结构化调试日志和离线验证支架先稳定下来，方便真正上机前复现问题。

## 当前已实现

- `metadata.yaml` / `_conf_schema.json` / `requirements.txt`
- 最小主链路：
  - `InputNormalizer`
  - `EligibilityChecker`
  - `BypassRouter`
  - `StateScopeResolver`
  - `StateRepository(SQLite)`
  - `ScopeLockManager`
  - `TurnRegistry`
  - `CommitWatchdog`
  - `StateDecay`
  - `StateResolver`
  - `ToolContextTracker`
  - `DialoguePolicy`
  - `LoreInjector`
  - `PromptMergePolicy`
  - `TokenBudgetGuard`
  - `PromptComposer`
  - `RuntimeInjectionGuard`
  - `FallbackController`
  - `ObservabilityLogger`
- 基础命令：
  - `/prhello`
  - `/prstate`
  - `/prturns`
- 离线开发支架：
  - `dev_harness.py`
  - `tests/`

## 调试日志

`DEBUG_MODE` 默认开启。开启后会输出结构化 JSON 日志，关键 turn 事件会带上这些字段：

- `turn_id`
- `user_scope_key`
- `session_scope_key`
- `bypass_reason`
- `streaming_flag`
- `selected_policy`
- `selected_lore_ids`
- `tool_flags`
- `commit_stage`
- `final_committed`
- `user_state_version`
- `session_state_version`

重点事件包括：

- `turn_created`
- `tool_entered`
- `candidate_write`
- `final_commit`
- `watchdog_rollback`

## 运行 dev_harness.py

先安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

运行全部离线链路：

```bash
python dev_harness.py
```

只运行某一条链路：

```bash
python dev_harness.py --scenario normal
python dev_harness.py --scenario tool
python dev_harness.py --scenario timeout
python dev_harness.py --scenario concurrent
```

可选链路：

- `normal`: `on_llm_request -> on_llm_response -> after_message_sent`
- `tool`: `on_llm_request -> on_agent_begin -> on_using_llm_tool -> on_llm_tool_respond -> on_agent_done -> after_message_sent`
- `timeout`: `on_llm_request -> on_llm_response -> 不触发 after_message_sent -> CommitWatchdog 超时回滚`
- `concurrent`: 同 session 两个并发 turn、同 user 跨两个 session 并发 turn
- `all`: 依次运行以上全部链路

`dev_harness.py` 不连接真实平台；它会注入最小 AstrBot mock，并使用临时 SQLite 数据库。插件核心模块、状态仓储、turn registry、watchdog 和提交逻辑仍然走真实代码。

## 运行 tests

使用标准库 unittest：

```bash
python -m unittest discover -s tests -v
```

当前测试覆盖：

- `StateRepository`
- `ScopeLockManager`
- `TurnRegistry`
- `CommitWatchdog`
- `ToolContextTracker`

## 建议上机前验证顺序

1. 本地运行 `python -m unittest discover -s tests -v`
2. 本地运行 `python dev_harness.py --scenario all`
3. 加载插件后执行 `/prhello`
4. 执行 `/prstate`
5. 普通聊天，确认 `on_llm_request` 被触发
6. 验证动态注入是否污染历史
7. 验证 `/help` 等命令是否绕过 LLM 链路
8. 验证工具调用钩子能否绑定到 `turn_id`
9. 验证 `after_message_sent` 是否稳定触发

## 当前未完成

- 更完整的 `StateResolver` 规则
- 真正的复杂 lore 评分/检索
- 更细的工具态策略
- `OutputRefiner`
- 上机后的平台差异适配

## 注意

- streaming 模式下，starter 目前只记录 `streaming_flag`，不扩展复杂流式后处理。
- `mood` 只在当前 turn 内现场计算，不进入 `SessionState`。
- `after_message_sent` 是首选最终提交点；若未触发，则由 `CommitWatchdog` 超时回滚候选态。
- `_conf_schema.json` 中已移除未使用的 `plugin_priority`，当前没有事件优先级接入逻辑。
