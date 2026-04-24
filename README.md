# astrbot_plugin_persona_runtime

AstrBot 单角色人格运行时插件。

当前仓库已经不是最早的 starter 骨架，而是一个可运行的分阶段实现版本，包含：

- 阶段 1：`scene -> behavior -> PromptPlan -> /prwhy`
- 阶段 2：示例库、示例选择、`/prexample`、`/prexamples`
- 阶段 3：`raw_turns`、`turn_traces`、`learning_buffer`
- 阶段 4：批学习分析、`weight_patches`、运行时 patch 生效、`/prlearn`、`/prpatches`
- 阶段 5：`deterministic_mode`、固定评估集、`/preval`、离线评估链路

## 当前能力

- 运行时人格注入
  - 输入归一化、命令绕过、作用域解析
  - 场景解析：`status_share`、`casual_chat`、`complaint`、`task_request`
  - 行为选择：`short_ack`、`followup_question`、`comfort`、`solution`
  - PromptPlan 驱动的 prompt 组装
- 示例库
  - 手动保存最近回复为示例
  - 按场景、标签、简单文本相似度选择示例
- 记录链
  - `raw_turns`
  - `turn_traces`
  - `learning_buffer`
- 学习链
  - 批量分析 `learning_buffer`
  - 生成 `behavior_weight_patch`
  - 生成 `example_tag_weight_patch`
  - patch 在行为概率和示例选择阶段生效
- 稳定性
  - `after_message_sent` 最终提交
  - watchdog 超时回滚
  - `deterministic_mode`
  - 固定评估集和离线 harness

## 命令

- `/prhello`
  - 插件连通性检查
- `/prstate`
  - 查看当前 user/session state
- `/prturns`
  - 查看 turn registry 快照
- `/prwhy`
  - 查看最近一轮 PromptPlan、场景、行为概率、模块、learning effects
- `/prexample last [N] scene=... tags=a,b`
  - 保存最近 1 轮或 N 轮已提交回复为示例
- `/prexample tag <id> scene=... tags=a,b`
  - 修改示例元数据
- `/prexample disable <id>`
  - 禁用示例
- `/prexample delete <id>`
  - 删除示例
- `/prexamples`
  - 列出最近示例
- `/prlearn`
  - 手动触发一次学习分析
- `/prpatches`
  - 查看已持久化 patch
- `/preval`
  - 运行固定评估集

## 关键配置

见 [_conf_schema.json](C:/Users/PC/.astrbot_launcher/instances/091cc1c1-7919-4d7c-8cea-c8c2dfa29f64/core/data/plugins/astrbot_plugin_persona_runtime/_conf_schema.json)。

常用项：

- `enable_runtime`
  - 是否启用运行时主链路
- `DEBUG_MODE`
  - 是否输出结构化调试日志
- `commit_timeout_seconds`
  - 候选提交等待 `after_message_sent` 的超时秒数
- `max_runtime_chars`
  - 单轮动态注入字符预算
- `static_persona_prompt`
  - 基础人格底盘 prompt
- `learning_min_batch_size`
  - 自动批学习触发的最小样本数
- `deterministic_mode`
  - 开启后跳过后台自动学习，便于稳定回归

## 数据文件

插件运行后会在 `data/` 下维护：

- `persona_runtime.db`
  - SQLite 主库
- `lore_keywords.json`
  - lore 关键词配置

当前数据库包含这些主要表：

- `user_states`
- `session_states`
- `examples`
- `raw_turns`
- `turn_traces`
- `learning_buffer`
- `weight_patches`

## 调试与验证

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

运行离线 harness：

```bash
python dev_harness.py --scenario all
```

可选场景：

- `normal`
- `tool`
- `timeout`
- `concurrent`
- `eval`
- `all`

## 建议验证顺序

1. 运行 `python -m unittest discover -s tests -v`
2. 运行 `python dev_harness.py --scenario all`
3. 加载插件后执行 `/prhello`
4. 发一条普通对话，随后执行 `/prwhy`
5. 保存一条示例：`/prexample last`
6. 触发几轮后执行 `/prlearn`
7. 查看 patch：`/prpatches`
8. 执行 `/preval`

## 当前边界

- 学习链仍然是规则批分析，不接专用 classifier/learner 模型
- 示例选择仍是轻量相似度，不是向量召回
- 评估集目前是固定 4 条回归样例，主要覆盖核心行为
- `deterministic_mode` 只关闭后台自动学习；手动 `/prlearn` 仍可执行
