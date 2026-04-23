# astrbot_plugin_persona_runtime (V3.2 starter)

这是一个和 `PROJECT_START_V3.2` 对齐的 AstrBot 插件 starter。

## 当前已实现

- `metadata.yaml` / `_conf_schema.json` / `requirements.txt`
- 最小主链：
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
- 基础指令：
  - `/prhello`
  - `/prstate`
  - `/prturns`

## 当前定位

这是 **starter**，不是最终成品。
已经搭好工程骨架、SQLite、两段提交骨架、turn 超时回滚和日志结构，但还有很多策略需要继续补：

- 更完整的 `StateResolver` 规则
- 真正的 lore 评分/检索
- 更细的工具态关联
- `OutputRefiner`
- 更完整的测试

## 建议验证顺序

1. 先加载插件，执行 `/prhello`
2. 执行 `/prstate`
3. 普通聊天，看是否触发 `on_llm_request`
4. 验证动态注入是否污染历史
5. 验证 `/help` 等命令是否会进入 LLM 链路
6. 验证工具调用钩子能否绑定到 `turn_id`
7. 验证 `after_message_sent` 是否稳定触发

## 注意

- streaming 模式下，starter **只跳过后处理**，不跳过人格运行层主链。
- `mood` 只在 `TurnContext` 中现场计算，不进入 `SessionState`。
- `after_message_sent` 是首选最终提交点；若未触发，由 `CommitWatchdog` 超时回滚候选态。
