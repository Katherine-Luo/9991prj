# LIDC-IDRI Baseline 仓库开发规则

## 1. 适用范围与事实来源

本文件位于仓库根目录，递归适用于整个仓库。仓库内不得创建与本文件重复或冲突的 `.AGENT.md`、`.AGENTS.md` 或嵌套 `AGENTS.md`。

项目有四个互补的事实来源：

1. `docs/PROJECT_STATUS.md`：唯一实时状态，并通过 `active_requirements` 和 `active_config` 指定当前协议。
2. `docs/PROTOCOL_INDEX.md`：协议版本、ACTIVE/SUPERSEDED 状态和允许用途。
3. 状态文档指定的 active requirements/config：当前科学协议、阶段需求和验收标准。
4. `AGENTS.md`：开发、审查、确认、提交和推送流程。

不得用状态文档或本文件改写科学协议。发现四者不一致时，停止相关实现，向用户报告冲突并等待明确指示。标记为 `SUPERSEDED` 的协议只允许用于历史 Bug、provenance 或复现审计，不得作为新开发依据。

## 2. 开发批次定义

“开发批次”是任何会创建、修改、移动或删除 Git tracked files 的任务，包括代码、测试、配置和文档修改。

以下操作不构成开发批次：

- 只读问答。
- 只读仓库检查或状态查询。
- 尚未落盘的讨论和规划。

状态同步审查对 `docs/PROJECT_STATUS.md` 的修订属于原开发批次，不单独递归触发新一轮双 agent 审查。

## 3. 每个开发批次开始前的强制读取

修改任何 tracked file 前必须依次完成：

1. 完整读取本文件。
2. 读取 `docs/PROJECT_STATUS.md` 的 YAML front matter，解析 `active_protocol`、`active_requirements` 和 `active_config`。
3. 完整读取 `active_requirements` 指定的需求文档，并读取 active config。
4. 核对 `docs/PROTOCOL_INDEX.md`，确认 active protocol 未被标为 `SUPERSEDED`。
5. 当 `operating_mode` 为 `NORMAL_DEVELOPMENT` 时，读取状态文档的“当前状态”“当前阶段”和“下一阶段”，直到 `NORMAL_READING_END`。
6. 当 `operating_mode` 为 `BUG_MAINTENANCE` 时，通读整份状态文档。
7. 检查 `git status`、当前分支和已有 diff，识别并保护用户的既有改动。
8. 从状态文档确定本批次唯一允许处理的阶段和对应验收标准。

只有在用户明确批准的 protocol migration 阶段，才允许 active requirements/config 处于待创建或待冻结状态；该临时状态必须写入状态文档，且迁移阶段完成前不得进入后续科学阶段。

遗漏任一步骤时，不得开始修改文件。

## 4. 阶段范围和顺序

正常开发只能处理 YAML 中的 `development_phase`。Bug 维护只能处理 `maintenance_phase` 所指向的 Bug 和必要验证，不得借维护工作开发其他阶段。

阶段生命周期固定为：

```text
NOT_STARTED
→ IN_PROGRESS
→ AWAITING_USER_APPROVAL
→ COMPLETED
```

`BLOCKED` 表示存在明确阻塞条件，可从 `IN_PROGRESS` 或 `AWAITING_USER_APPROVAL` 进入；解除阻塞后返回相应的未完成状态。

必须遵守：

- 不得提前实现、搭建、迁移或详细规划未来阶段。
- 跨阶段依赖只有在当前阶段需求明确要求时才能实现，并仍归入当前阶段验收。
- 当前阶段技术验收及阶段级双 agent 审查通过后，状态只能先改为 `AWAITING_USER_APPROVAL`。
- 用户明确确认后，才能把当前阶段标记为 `COMPLETED`。
- 当前阶段确认完成并推送后，才允许制定下一阶段实施计划。
- 下一阶段计划得到用户明确批准后，才能把下一阶段改为 `IN_PROGRESS` 并开始开发。
- 如果用户请求的实现属于未来阶段，先说明当前阶段门限制，不执行该实现。

## 5. 冻结需求文档保护

未经用户针对具体变更的明确批准，不得对任何已冻结 requirements/config 执行编辑、格式化、重命名、移动或覆盖。

即使科学协议变更得到批准，也必须创建新协议版本并保留旧版本，不得静默覆盖。Baseline-v1 的 requirements/config/resolved config/hash 永久保持历史只读。

每个开发批次结束时必须验证：

```bash
git diff --exit-code -- \
  docs/LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md \
  configs/baseline_v1.yaml \
  configs/baseline_v1.resolved.yaml \
  configs/baseline_v1.sha256
```

该命令非零退出时，本批次不得声明完成或提交。

## 6. 语言规范

- 所有面向用户的回答使用中文。
- `AGENTS.md`、需求文档、协议索引和项目状态文档使用中文。
- 源代码、标识符、类型名、函数名、测试名、配置键、注释、docstring 和面向开发者的代码消息使用英语。
- 不得为了语言统一而翻译或重写冻结需求文档。
- Git commit subject 使用简洁英语，并延续仓库现有风格。

## 7. 开发与验证流程

每个开发批次按以下顺序执行：

1. 完成强制读取和工作区检查。
2. 将本批次范围映射到当前阶段 requirement ID 和验收标准。
3. 只实现当前阶段所需的最小原子改动。
4. 为行为变更添加或更新直接测试；纯文档改动执行结构、内容、链接和 whitespace 检查。
5. 运行与改动相称的完整验证，并保存命令和结果。
6. 串行执行“阶段合规审查”和“状态同步审查”。
7. 独立复核最终 diff、测试结果和需求文档无改动证明。
8. 审查通过后，按原子提交规则创建本地 commit。

测试失败、验收证据不足、越阶段、需求文档出现未批准改动或审查 agent 不可用时，必须停止完成流程并如实报告阻塞。

## 8. 第一审查：Phase Compliance Reviewer

每个开发批次结束时必须调用一名独立的 `Phase Compliance Reviewer`。该 agent 只能进行只读审查，不得修改任何文件。

审查范围：

- Active requirements、active config、协议索引和全部历史冻结协议。
- 状态文档及当前阶段。
- 本批次 diff 和相关代码、测试、配置或文档。
- 已执行的验证命令及结果。
- 是否存在未来阶段实现或无关改动。

审查报告必须包含：

```text
Verdict: PASS | FAIL
Current phase:
Requirements reviewed:
Acceptance criteria evidence:
Findings:
Out-of-phase check:
Blocking gaps:
```

规则：

- 每条 finding 必须关联 requirement ID、验收标准或本文件规则。
- 存在任一阻断缺口时必须给出 `FAIL`。
- 主 agent 修复 `FAIL` 后，必须重新调用独立合规审查，直到获得 `PASS`。
- 不得仅凭 agent 的结论宣称完成；主 agent 仍需自行验证证据。

## 9. 第二审查：Status Synchronization Reviewer

合规审查通过后，必须串行调用另一名独立的 `Status Synchronization Reviewer`。

该 agent 必须比较：

- 当前代码、测试、配置和文档。
- Git 分支、HEAD、status 和 diff。
- 完整 `docs/PROJECT_STATUS.md`。
- 当前批次验证与合规审查结果。

必须核对：

- 工作模式、当前阶段和下一阶段。
- 已完成、正在进行和尚未完成的工作。
- 阶段验收证据和阶段门结论。
- 活动 Bug、未解决困难及受影响阶段。
- 产物路径、commit 证据和最近更新时间。

若状态不同步，该 agent 只允许修改 `docs/PROJECT_STATUS.md`，不得修改代码、测试、配置、`AGENTS.md` 或冻结需求文档。

审查报告必须包含：

```text
Verdict: IN_SYNC | UPDATED | BLOCKED
Current phase:
Compared evidence:
Status mismatches:
Status updates made:
Remaining blockers:
```

主 agent 必须检查该 agent 的实际 diff，不能只信任报告。状态仍不同步或 agent 无法运行时，本批次不得声明完成。

## 10. 阶段门与用户确认

当当前阶段全部验收标准满足时：

1. 运行覆盖整个阶段的完整验证。
2. 再运行一次阶段级 `Phase Compliance Reviewer`。
3. 再运行一次阶段级 `Status Synchronization Reviewer`。
4. 将阶段状态设为 `AWAITING_USER_APPROVAL`，下一阶段保持 `NOT_STARTED`。
5. 用中文向用户报告完成内容、验收证据、开放困难、Bug 和本地 commits。
6. 明确请求用户确认本阶段，不得把沉默视为批准。
7. 收到明确确认后，将阶段状态单独更新并提交为 `COMPLETED`。
8. 验证本地分支、commit 范围、remote 和目标分支后，才可推送该阶段 commits。
9. 推送成功后，才开始制定下一阶段实施计划。

P0–P10 任一阶段均不得跳过该确认门。

## 11. Git 原子提交规则

一个 commit 只包含一个可独立审查和回退的原子任务。

- 一个功能及其直接测试可以属于同一原子 commit。
- 状态文档同步必须使用单独 commit。
- 不得在一个 commit 中混合无关功能、重构、修复或文档改动。
- 必须显式暂存目标路径，禁止使用 `git add .` 或 `git add -A`。
- 提交前使用 `git diff --cached --name-only` 和 `git diff --cached` 审查暂存内容。
- 不得提交或清理用户已有的无关改动。
- 不得修改、压缩、重写或删除用户 commits，除非用户明确要求。

阶段内允许在双 agent 审查通过后创建本地原子 commits，但未经阶段确认不得 push。

## 12. 推送规则

- `git push` 是阶段确认后的独立动作，不因“请实现”而自动获得授权。
- 当前阶段未标记 `COMPLETED` 时不得推送该阶段 commits。
- 推送前必须确认用户已明确批准本阶段、工作区状态符合预期、分支和 remote 正确。
- 不得使用 force push，除非用户对具体目标明确授权且已说明风险。
- 推送完成后验证 local HEAD、upstream 和 remote branch commit 一致。

## 13. Bug 维护

发现历史阶段 Bug 时，严格按 `docs/PROJECT_STATUS.md` 切换到 `BUG_MAINTENANCE / FULL_DOCUMENT`：

- 保存 `resume_phase`。
- 设置 `maintenance_phase` 和 `active_bug_ids`。
- 只修复登记 Bug 及其必要回归测试。
- 根据验收是否失效，将相关阶段标记为 `AT_RISK` 或 `INVALIDATED`。
- 修复后记录根因、修改、验证和 commit。
- 必要时重新运行受影响阶段门和双 agent 审查。
- 用户确认修复及重新验收结果后，才能恢复正常开发并推送。

不得删除历史 Bug、困难或阶段完成记录。

## 14. 完成声明标准

只有同时满足以下条件，才可称一个开发批次“完成”：

- 改动仅属于允许的当前阶段或维护阶段。
- 对应测试和验证使用最终代码重新运行并通过。
- `Phase Compliance Reviewer` 给出 `PASS`。
- `Status Synchronization Reviewer` 给出 `IN_SYNC` 或完成必要更新后无剩余阻塞。
- 主 agent 已检查最终 diff 和 Git 状态。
- Active requirements/config 与全部历史冻结协议没有未批准改动。
- 状态文档准确反映代码、测试、困难、Bug 和验收状态。

开发批次完成不等于阶段完成。阶段只有经过完整阶段门、进入 `AWAITING_USER_APPROVAL` 并获得用户明确确认后，才能标记为 `COMPLETED`。
