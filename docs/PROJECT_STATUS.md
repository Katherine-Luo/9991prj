---
schema_version: 1
project: LIDC-IDRI Baseline-v1
operating_mode: NORMAL_DEVELOPMENT
reading_scope: CURRENT_AND_NEXT
development_phase: P2
development_phase_status: COMPLETED
maintenance_phase: null
active_bug_ids: []
resume_phase: P2
next_phase: P3
last_updated: 2026-08-09
last_verified_commit: d3f3995; P2 completion state commit pending
---

# LIDC-IDRI Baseline-v1 项目状态

本文件是项目开发状态的唯一事实来源。科学协议以[冻结需求文档](./LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md)为准；状态文件只记录实施进度、Bug、困难和阶段门结果，不复制或修改科学协议。

## 1. 阅读规则

首先读取顶部 YAML 状态，再按以下规则决定阅读范围：

- `NORMAL_DEVELOPMENT`：阅读“当前状态”“当前阶段”和“下一阶段”，到 `NORMAL_READING_END` 标记即可停止。
- `BUG_MAINTENANCE`：必须通读整份文档，重点检查完整阶段总表、活动 Bug、未解决困难、受影响的下游阶段和历史阶段记录。
- `reading_scope` 必须与模式一致：正常开发为 `CURRENT_AND_NEXT`，Bug 维护为 `FULL_DOCUMENT`。

## 2. 当前状态

| 字段 | 当前值 |
|---|---|
| 工作模式 | `NORMAL_DEVELOPMENT` |
| 阅读范围 | `CURRENT_AND_NEXT` |
| 当前开发阶段 | `P2 Physical nodule cohort` |
| 阶段状态 | `COMPLETED` |
| 维护目标阶段 | 无 |
| 活动 Bug | 无 |
| 当前阻塞项 | 无；用户已明确确认 P2，阶段封存与交付进行中 |
| 恢复阶段 | `P2` |
| 下一阶段 | `P3 Consensus mask 与 ROI` |
| 最近更新 | 2026-08-09 |
| 状态依据 | `c1c1b95` 保存 P2 implementation/tests/.gitignore，`28e46b1` 保存 P2 deidentified audit evidence，`d3f3995` 保存待确认状态；用户已确认 P2，正在创建单独封存状态 commit，随后合并并推送 |

## 3. 当前阶段：P2 Physical nodule cohort

### 阶段目标

将 canonical XML 的 `nodule >=3 mm` reader annotations 与 pylidc spatial clusters 对齐，建立 stable source-derived `nodule_uid`、reader aggregation 和本地私有 cohort manifest。

### 已完成

- P1 的 canonical XML、CT geometry、series eligibility 和 4 个不同图像内容 series 的排除决定均已确认并推送。
- P2 实施计划已获用户批准：使用 pylidc 默认 clustering rule 与最小 `np.int` runtime compatibility adapter；完整 manifest 本地私有，Git 仅保存脱敏审计证据。
- strict computed `>3 mm` flag 固定为 cluster 内 reader axial diameter 的最大值严格大于 `3.0 mm`；不影响 primary inclusion。
- 已实现 canonical XML source parser、pylidc matching/clustering、annotation-to-cluster mapping、stable source-derived `nodule_uid`、reader aggregation、私有 manifest schema validation 和脱敏审计报告。
- 完整本地 cohort audit 已在 1,014 个 P1-eligible CT series 上完成：2,634 个 physical clusters；1,073 个 binary primary nodules、578 位 binary-cohort patients；`>=3 readers` sensitivity cohort 为 438 个 nodules。另有 1,560 个 uncertain malignancy clusters 与 1 个 missing-required-target cluster，均保留审计记录且不进入 primary binary cohort。
- P1 已排除的 4 个 series 继续排除；14 个超过 4 readers 的 clusters 记录为排除；primary inclusion 始终以 XML `nodule >=3 mm` annotation class 决定，而非 computed diameter。与 reference 2,651 nodules / 875 patients 的差异仅在 reconciliation 中报告，`hard_gate=false`。
- 私有 `artifacts/manifests/nodules.parquet` 与 `annotation_mapping.parquet` 已生成且保持 Git ignore；前者含 2,634 个唯一 `nodule_uid`、source XML/DICOM SOP fingerprints、annotation class、diagnostic-only pylidc SQL IDs、9 个逐 target valid-reader counts、原始/聚合 targets、tie flags 和 strict-diameter flag。
- 已生成可提交的脱敏审计证据：`artifacts/audit/p2/summary.json`、`reconciliation.csv`、`exclusions.csv`、`clustering_tolerances.csv`。全部仅使用 SHA-256 derived identifiers，不含原始 UID、patient ID 或绝对路径。
- 每-series clustering tolerance 已保存 1,014 行；`tol=None` 按 scan slice thickness 解析，effective tolerance 范围为 `0.10131387882547446–5.0 mm`。
- 所有 64 项自动测试通过；P2 阶段级 Phase Compliance Reviewer 已给出 `PASS`。

### 正在进行

- 无 P2 开发工作；用户已确认，正在执行状态封存、合并与 GitHub 推送。

### 尚未完成

- P2 completion state commit、`main` fast-forward merge 与 GitHub push；此为交付动作，不属于 P3 开发。
- P3 保持 `NOT_STARTED`；在 P2 推送成功后才允许制定其实施计划。

### 验收进度

| P2 验收项 | 状态 | 证据 |
|---|---|---|
| P2-R1 primary cohort 与 reconciliation | `PASS` | XML class-selected canonical source、1,014 P1-eligible series、2,634 physical clusters、1,073 binary primary nodules / 578 patients；2,651/875 仅 reconciliation，`hard_gate=false` |
| P2-R2 stable provenance 与 `nodule_uid` | `PASS` | `nodule_uid` 由 patient/study/series source identifiers、canonical XML hash 和排序 source fingerprints 导出；SQL ID independence、source/SOP-content sensitivity、mapping determinism 与 duplicate UID checks 通过 |
| P2-R3 reader aggregation | `PASS` | 9 个逐 target valid-reader counts、raw ratings、normalized continuous targets、categorical soft vote distributions、reader count、reader dispersion、`>=3 readers` 和 strict diameter flag 已写入 private manifest；缺失 target 具体记录 |
| P2-R4 categorical ties | `PASS` | all target-ready clusters 的 internalStructure/calcification modal ties 分别为 8/55；binary primary subset 为 2/46；ties 保留在 soft-target cohort，未从 primary cohort 删除 |
| P2 自动测试、双 agent 审查与用户确认 | `PASS` | `64 passed`；Phase Compliance Reviewer `PASS`、Status Synchronization Reviewer `UPDATED`；用户已明确确认 P2 |

### 未解决困难

- `DIF-P10-001` 继续为 `OPEN`，不影响 P2 本地 cohort 开发。
- P2 无开放技术困难或待确认阶段门。

## 4. 下一阶段：P3 Consensus mask 与 ROI

### 阶段目标

对已确认的 P2 physical nodules 生成 50% consensus mask、固定 `64³` ROI 和 QA 证据。

### 进入条件

- P2 cohort、stable provenance、reader aggregation 和 reconciliation 通过阶段门并取得用户确认。

### 第一批任务

- 尚未制定；P3 保持 `NOT_STARTED`。

<!-- NORMAL_READING_END -->

---

## 5. P1 历史阶段记录

### 阶段目标

确认 `LIDC-XML-only` 为唯一 canonical annotation source，盘点 CT/DX/CR/CXR 数据，建立 XML–DICOM UID mapping，并验证 CT series 的几何完整性和确定性空间切片排序。

### 已完成

- P0 已完成、确认并推送；冻结配置与 P0 环境/审计证据可复用。
- P1 实施计划已获用户明确批准；审计产物采用精简提交策略：提交脱敏摘要、每-series 报告与异常清单，不提交逐切片明细。
- 已确认本地原始数据根目录为 `/Users/katherine/Desktop/lidc_data`，其中 canonical XML 与 DICOM 下载树均存在。
- 已实现 header-only P1 audit、合成 XML/DICOM fixtures、确定性与脱敏报告测试。
- 完整本地审计已生成脱敏的 `summary.json`、`series_audit.csv` 和 `anomalies.csv`：1,010 patients、1,018 CT series、243,958 CT instances、513 DX、56 CR；1,035/1,035 canonical CT XML 均唯一映射到 CT series，且 1,018/1,018 CT series 均被 canonical CT XML 覆盖。
- canonical XML 合计为 1,318，较本地参考盘点 1,319 少 1；该 reconciliation 的 `hard_gate=false`，不构成失败。
- 已对 5 个阻断性异常 series 完成只读像素内容复核。唯一的完全相同 duplicate slice 仅在未来构建派生 CT volume 时按最小 SOP UID 确定性保留一张；原始 DICOM 没有删除、修改或覆盖。
- 另外 4 个 duplicate plane 图像内容不同的 CT series 已按用户批准的 Baseline-v1 决定整 series 排除，未从中任意选择切片。排除的匿名 `series_key` 为 `02b556d221dfcd678936f9e12bed22d3a9b17905021620f6093149f420d6f99d`、`cc2f1f6389084459fc15040649fdd47da2c50b2ce9d41d7d08185090bf9f8e65`、`ee8fcb4cf275c52e977f153e2d907f25918d3a6cd772c707298acae90132cf5e`、`fed15fa97f94e5b59a8c59cb046ef7d710ca23826a1a70173d118ea69317c91e`。
- 该排除涉及 4 个 patients 和 63 条 `unblindedReadNodule`（`nodule >=3 mm`）reader annotations。该 63 是 P1 annotation-level 计数，**不是** physical-nodule count；physical-nodule 聚类和计数属于 P2，尚未执行。
- 处理后有 1,014 个 CT series 可进入后续 pipeline；其中 exact-duplicate series 的派生 volume selection 保存在独立脱敏产物中。
- 已重新运行 P1 audit 和 approved resolution；P1 专项测试为 `7 passed`，完整测试为 `44 passed`。
- 阶段级 Phase Compliance Reviewer 已给出 `PASS`，确认 P1-R1/P1-R2 证据完整、无越阶段实现。
- 用户已明确确认 P1 阶段结果；本阶段现在封存为 `COMPLETED`。

### 正在进行

- 无 P1 开发工作。P2 仍为 `NOT_STARTED`，在单独制定并获得用户批准的 P2 实施计划前不得开始。

### 尚未完成

- P1 无未完成技术或验收项。P2 的实施计划、批准和开发均不属于本阶段。

### 验收进度

| P1 验收项 | 状态 | 证据 |
|---|---|---|
| P1-R1 canonical XML 与 UID mapping | `PASS` | `summary.json`：1,035/1,035 canonical CT XML mapped，1,018/1,018 CT series covered；CT/DX/CR/CXR 与 embedded XML 均单独统计 |
| P1-R2 DICOM geometry | `PASS` | 空间投影排序和异常检测已完成；原始 `anomalies.csv` 保留 13 个 `DUPLICATE_SLICE_PLANE`、4 个 `SPACING_NONUNIFORM` 与 2 个 `SUSPECTED_SLICE_GAP`。resolution policy 将 4 个不同内容 series 排除、对 1 个完全相同 duplicate 记录派生 volume selection；剩余 1,014 CT series 可确定性进入后续 pipeline |
| P1 自动测试 | `PASS` | P1 专项 `7 passed`；完整套件 `44 passed` |
| Phase Compliance Reviewer | `PASS` | 阶段级审查确认 P1-R1/P1-R2、用户批准的 duplicate-plane policy、脱敏产物、只读原始数据及无越阶段实现 |
| Status Synchronization Reviewer | `UPDATED` | 状态已同步 resolution 实现、证据、阶段级审查与用户确认后的 `COMPLETED` |

P1 技术阶段门和用户确认均已通过，阶段封存为 `COMPLETED`。`anomalies.csv` 是对原始数据的审计记录，不会因 resolution 而被改写；后续 eligibility 仅以 `duplicate_plane_resolution.json`、`duplicate_plane_resolution.csv` 和 `derived_volume_selection.csv` 为准。P2 仍为 `NOT_STARTED`，不得自动进入。

### 未解决困难

- `DIF-P10-001` 继续为 `OPEN`，但不影响 P1 的本地 header-only audit。

### 当前证据与产物

- [冻结需求文档](./LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md)
- [仓库开发规则](../AGENTS.md)
- P1 分支：`p1-dicom-xml-audit`；`c70e068` 保存 duplicate-plane resolution 实现与测试，`7d34cbf` 保存脱敏 resolution 证据，`fa73a1b` 与 `f4d1964` 保存 P1 状态同步。用户已确认 P1；本次阶段封存状态提交后执行推送。
- 原始数据：`/Users/katherine/Desktop/lidc_data`；初始 audit 仅读取 DICOM headers。已批准的 duplicate-plane 复核只读取 5 个目标 series 的 duplicate groups 像素数据以判定内容是否完全相同，且没有修改任何原始文件。
- P1 初始 audit 输出：`artifacts/audit/p1/summary.json`、`series_audit.csv`、`anomalies.csv`（脱敏）。`anomalies.csv` 是原始数据异常记录。
- P1 resolution 输出：`artifacts/audit/p1/duplicate_plane_resolution.json`、`duplicate_plane_resolution.csv`、`derived_volume_selection.csv`（均脱敏）。这些文件记录最终的 P1 后续可用性决定；可选逐切片明细保持本地 ignored。

## 6. P2 启动前历史快照

本节记录 P1 完成后、P2 实施计划获批前的状态，保留用于审计。它已于 2026-08-09 被第 3 节的 P2 `IN_PROGRESS` 当前状态取代；不得将本节的 `NOT_STARTED` 表述理解为当前阶段状态。

### 阶段目标

基于已审计的 canonical XML 与 CT volumes，将 reader annotations 聚类为 physical nodules，并建立 Baseline-v1 primary cohort、reader aggregation 和 stable provenance。

### 进入条件

- P1 canonical XML、XML–DICOM mapping 与 CT geometry 审计通过并已获用户确认。
- `DIF-P1-001` 已解决或形成明确审阅结论，并完成所需复核审计。

### 第一批任务

- 当时尚未制定；P2 当时保持 `NOT_STARTED`。

### 已知风险

- P2 不能绕过 P1 的 canonical source、mapping 或 geometry 审计结论。

以下为完整历史与维护区。正常开发模式无需每次通读；`BUG_MAINTENANCE` 模式必须从此处继续阅读到文件末尾。

## 5. 状态模型与更新规则

### 阶段生命周期

| 状态 | 含义 |
|---|---|
| `NOT_STARTED` | 尚未进入该阶段 |
| `IN_PROGRESS` | 正在开发或重新验收 |
| `AWAITING_USER_APPROVAL` | 技术验收和双 agent 审查已通过，正在等待用户明确确认 |
| `BLOCKED` | 阶段无法继续，且已有明确阻塞条件 |
| `COMPLETED` | 全部阶段门验收已通过、用户已明确确认并保存证据 |

正常阶段流转固定为：

```text
NOT_STARTED
→ IN_PROGRESS
→ AWAITING_USER_APPROVAL
→ COMPLETED
```

下一阶段在其实施计划获得用户明确批准前保持 `NOT_STARTED`。`BLOCKED` 用于明确的技术或外部阻塞，不得用于表示正常等待用户确认。

### 阶段健康状态

| 状态 | 含义 |
|---|---|
| `NOT_APPLICABLE` | 尚未开始，暂无健康判断 |
| `ON_TRACK` | 当前没有已知问题使验收失效 |
| `AT_RISK` | 活动 Bug 或困难可能影响该阶段，但尚未确认验收失效 |
| `INVALIDATED` | 已确认原验收结果失效，必须重开和重新验收 |

### 正常开发更新

每个有效开发批次结束时：

1. 更新 YAML 中的阶段、模式、下一阶段和最近更新时间。
2. 更新“当前阶段”的已完成、正在进行、尚未完成、验收证据和困难。
3. 串行调用独立的阶段合规审查 agent 和状态同步审查 agent；合规审查只读，状态同步审查如发现差异只能修改本状态文档。
4. 状态同步使用独立的原子 Git commit，不与实现、测试或修复 commit 混合；不复制普通 Git commit 日志。
5. 阶段内允许在双 agent 审查通过后创建本地原子 commits，但未经阶段确认不得推送。

当阶段门的技术验收全部通过时：

1. 运行覆盖整个阶段的双 agent 审查。
2. 将当前阶段标记为 `AWAITING_USER_APPROVAL`，保存验收证据，下一阶段继续保持 `NOT_STARTED`。
3. 向用户报告阶段结果并等待明确确认，不得把沉默视为批准。
4. 用户确认后，使用独立状态 commit 将当前阶段标记为 `COMPLETED` 并追加永久阶段记录。
5. 验证 branch、remote 和 commit 范围后，推送该阶段的本地原子 commits。
6. 推送成功后才制定下一阶段实施计划；该计划获得用户批准后，下一阶段才切换为 `IN_PROGRESS`。

### Bug 维护模式

发现阶段 Bug 时：

1. 将 `operating_mode` 改为 `BUG_MAINTENANCE`，`reading_scope` 改为 `FULL_DOCUMENT`。
2. 保留原前向开发阶段为 `resume_phase`，并设置 `maintenance_phase` 和 `active_bug_ids`。
3. 在 Bug 登记表创建 `BUG-P{phase}-{NNN}` 记录。
4. 若 Bug 使原阶段验收标准失效，将该阶段生命周期改为 `IN_PROGRESS` 或 `BLOCKED`，健康状态改为 `INVALIDATED`。
5. 若 Bug 未使验收失效，原阶段保持 `COMPLETED`，但在修复期间健康状态改为 `AT_RISK`。
6. 受影响的下游阶段标记为 `AT_RISK`，但不删除其原完成记录。

Bug 修复后：

1. 记录根因、修改内容、验证命令、验证结果和修复 commit。
2. 对失效的阶段重新执行阶段门。
3. 将 Bug 标记为 `RESOLVED`；只有重新验收通过后才能恢复阶段的 `COMPLETED / ON_TRACK`。
4. 运行阶段合规审查和状态同步审查，并向用户报告修复及重新验收结果。
5. 用户明确确认后，清除 `maintenance_phase` 和 `active_bug_ids`，恢复 `NORMAL_DEVELOPMENT / CURRENT_AND_NEXT`，回到 `resume_phase`，然后才可推送修复 commits。

### 未解决困难

困难不得因阶段切换而删除。其状态只能在以下值间变化：

- `OPEN`
- `MITIGATED`
- `RESOLVED`

## 6. 完整阶段总表

| 阶段 | 名称 | 生命周期 | 健康状态 | 阶段门 | 开放 Bug | 开放困难 |
|---|---|---|---|---|---:|---:|
| P0 | 工程环境与配置冻结 | `COMPLETED` | `ON_TRACK` | `PASS` | 0 | 0 |
| P1 | DICOM/XML 审计 | `COMPLETED` | `ON_TRACK` | 技术验收、阶段级双 agent 审查和用户确认均为 `PASS` | 0 | 0 |
| P2 | Physical nodule cohort | `COMPLETED` | `ON_TRACK` | P2-R1–P2-R4、自动测试、双 agent 审查和用户确认均为 `PASS`；P3 保持未开始 | 0 | 0 |
| P3 | Consensus mask 与 ROI | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P4 | Patient-level split 与共享初始化 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P5 | Black-box DenseNet | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P6 | Standard CBM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P7 | Mixed-type CEM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P8 | CBM + GAM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P9 | 统一评估 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P10 | Katana 正式实验与报告 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 1 |

## 7. Bug 登记表

### 活动 Bug

当前无活动 Bug。

### Bug 状态

`OPEN` → `INVESTIGATING` → `FIXING` → `VERIFYING` → `RESOLVED`

### Bug 记录模板

后续 Bug 必须按以下字段追加，禁止覆盖已经关闭的记录：

```markdown
### BUG-P{phase}-{NNN}：简短标题

- 状态：OPEN | INVESTIGATING | FIXING | VERIFYING | RESOLVED
- 严重度：CRITICAL | HIGH | MEDIUM | LOW
- 发现日期：YYYY-MM-DD
- 影响阶段：P?
- 影响验收标准：是 | 否
- 恢复阶段：P?
- 受影响下游阶段：[]
- 现象：
- 复现方式：
- 根因：
- 修复：
- 验证命令与结果：
- 未解决事项：
- 修复 commit：
```

## 8. 未解决困难登记表

### DIF-P2-001：pylidc 与锁定 NumPy 的 `np.int` compatibility defect

- 状态：`RESOLVED`
- 所属阶段：P2
- 首次记录：2026-08-09
- 解决日期：2026-08-09
- 影响：直接调用 pylidc 0.2.3 的 contour/clustering 代码会在 NumPy 1.26.4 因已移除的 `np.int` 失败，阻止 physical nodule clustering。
- 当前结论：最小运行时 adapter 仅在 alias 缺失时设置 `np.int = int`，未修改 pylidc 源码、SQLite database、依赖锁定或原始数据。
- 解决证据：固定 default clustering 参数、effective-tolerance reconstruction、source mapping determinism 与 complete P2 cohort audit 均通过；`summary.json` 记录 `np_int_compatibility_enabled=true`，全量测试 `64 passed`。
- 解除条件：已满足；P2 full audit 成功，且无原始数据、数据库或依赖文件修改。
- 关联 Bug：无；这是第三方已锁定依赖的 compatibility defect。

### DIF-P1-001：CT series 重复 slice plane 的处理结论

- 状态：`RESOLVED`
- 所属阶段：P1
- 首次记录：2026-08-09
- 解决日期：2026-08-09
- 原影响：P1-R2 的“每个纳入 CT series 能确定性构建 3D volume”验收无法通过；P2 不得开始。
- 当前结论：用户已明确批准以下可追溯 policy，并已完成复核和实现：(1) 对唯一完全相同的 duplicate slice，仅在构建派生 CT volume 时确定性保留 SOP UID 字典序最小的一张，原始 DICOM 不修改；(2) 对 4 个同一空间位置但图像内容不同的 series，整 series 从 Baseline-v1 后续 pipeline 排除，不自动选择其中一张。P1 的 `anomalies.csv` 继续保留原始 13 条 duplicate-plane 审计发现；resolution 输出独立记录最终 eligibility。
- 解决证据：`duplicate_plane_resolution.json` 记录 1 exact-duplicate eligible series、4 different-content excluded series、4 affected patients、63 条 `nodule >=3 mm` reader annotations、1,014 eligible CT series 和 0 个 raw DICOM 文件修改；`derived_volume_selection.csv` 记录唯一的确定性保留选择。63 是 annotation-level 计数，**不是** physical-nodule count；后者必须在 P2 使用 clustering 后生成。P1 专项测试 `7 passed`，完整测试 `44 passed`，阶段级 Phase Compliance Reviewer 为 `PASS`。
- 解除条件：已满足；P1 已完成，P2 仍未开始。
- 关联 Bug：无；这是原始数据审计发现，尚非实现 Bug。

### DIF-P0-001：Katana CUDA 环境尚未验证

- 状态：`RESOLVED`
- 所属阶段：P0
- 首次记录：2026-08-08
- 解决日期：2026-08-09
- 原影响：在正式训练环境完成 CUDA forward/backward smoke test 前，P0 阶段门不能通过。
- 当前结论：Katana CUDA 环境、依赖、存储和 PBS batch smoke 已全部验证，P0 不再受此困难阻塞。
- 解决证据：PBS job `8942735.kman.restech.unsw.edu.au` 在 NVIDIA L40S 上 Exit 0；`artifacts/audit/p0/cuda.json` 与 `katana-linux-environment.json` 均为 `PASS`；配置哈希为 `6a17fd6f3731eb3307cf296fb203e58cf35adb08c3cfd984b136424421fe4a1c`。
- 解除条件：已满足。
- 关联 Bug：无。

### DIF-P10-001：Katana user scratch 扩容申请等待回复

- 状态：`OPEN`
- 所属阶段：P10
- 首次记录：2026-08-08
- 影响：正式训练、Grad-CAM 和中间产物可能超过当前 scratch 的安全容量。
- 当前结论：不阻塞 P0。P0 只使用合成 ROI，远程工作集控制在 20 GB 以内。
- 缓解措施：不上传原始 DICOM；通过 KDM 传输；正式 job 使用 `$TMPDIR`；重要数据和证据保留本地副本。
- 下一步：等待扩容回复；P3 完成后测量 ROI 大小并估算正式训练和报告产物的总工作集。
- 解除条件：Katana 可用存储不少于预计正式工作集的 120%，或学校批准足够的扩容空间。
- 关联 Bug：无。

## 9. 阶段永久记录

### P0 完成记录

- 完成日期：2026-08-09
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：工程骨架、依赖与双平台环境锁定、机器可读配置及 SHA-256、固定 seed、CPU/MPS/CUDA DenseNet smoke、Katana PBS 与存储 gates、本地 37 项测试、Phase Compliance Reviewer、Status Synchronization Reviewer 和用户阶段确认。
- 明确未纳入内容：P1 DICOM/XML 审计及任何后续阶段实现；P1 保持 `NOT_STARTED`。
- 验收标准与证据：配置哈希 `6a17fd6f3731eb3307cf296fb203e58cf35adb08c3cfd984b136424421fe4a1c`；CPU/MPS/CUDA audit 均为 `PASS`；PBS job `8942735.kman.restech.unsw.edu.au` Exit 0；Katana 工作集 `7,543,988,928` bytes；本地 `37 passed`；两位独立 reviewer 均为 `PASS`。
- 产物路径：`configs/`、`environment/`、`artifacts/audit/p0/`、`scripts/katana/`、`tests/`。
- 已解决 Bug：无。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`，不影响 P0 完成结论。
- 阶段门结论：`PASS`
- 完成 commit：`a16e3b5`。
- 交付状态：`a16e3b5` 已 fast-forward 合并至 `main` 并推送 GitHub，远程 SHA 核对一致；合并后的完整测试为 `37 passed`。
- 清理状态：Katana P0 临时目录和代码缓存已删除；Conda package cache 从 `899 MB` 降至 `629 MB`，scratch 总占用为 `6.5 GB`；保留 `5.7 GB` 已验证环境和 `2.9 GB` pip 缓存。本机约 `148 MB` 临时副本及 P0 worktree 已删除，测试、审计证据、原始数据、本机环境和 `.DS_Store` 均保留。

### P1 完成记录

- 完成日期：2026-08-09
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：确认 `LIDC-XML-only` 为唯一 canonical XML source；完成 CT/DX/CR/CXR inventory、canonical XML–CT Series UID mapping、DICOM header geometry audit、确定性空间投影排序、异常报告和脱敏审计产物。所有 1,035 份 canonical CT XML 均唯一映射，1,018 个 CT series 均被 canonical CT XML 覆盖。
- 已批准 duplicate-plane 处理：对唯一完全相同的 duplicate slice，仅在派生 CT volume 构建时按 SOP UID 字典序保留一张；原始 DICOM 不删除、不修改。对同一空间位置但图像内容不同的 4 个 CT series 整 series 从 Baseline-v1 后续 pipeline 排除，不选择其中任意一张 slice。
- 最终排除的匿名 CT `series_key`：`02b556d221dfcd678936f9e12bed22d3a9b17905021620f6093149f420d6f99d`、`cc2f1f6389084459fc15040649fdd47da2c50b2ce9d41d7d08185090bf9f8e65`、`ee8fcb4cf275c52e977f153e2d907f25918d3a6cd772c707298acae90132cf5e`、`fed15fa97f94e5b59a8c59cb046ef7d710ca23826a1a70173d118ea69317c91e`。
- 影响范围：排除涉及 4 位患者和 63 条 `nodule >=3 mm` reader annotations；63 是 P1 annotation-level 计数，**不是** physical-nodule count。P2 的 clustering 尚未执行。最终 1,014 个 CT series 可进入后续 pipeline。
- 明确未纳入内容：physical nodule clustering、cohort/标签生成、consensus mask、ROI、split 或模型开发；P2 保持 `NOT_STARTED`。
- 验收标准与证据：P1-R1/P1-R2 `PASS`；`summary.json`、`series_audit.csv`、`anomalies.csv`、`duplicate_plane_resolution.json`、`duplicate_plane_resolution.csv` 和 `derived_volume_selection.csv`；P1 专项测试 `7 passed`、完整测试 `44 passed`；Phase Compliance Reviewer `PASS`、Status Synchronization Reviewer `UPDATED`。
- 产物路径：`artifacts/audit/p1/`、`src/lidc_baseline/p1_audit.py`、`src/lidc_baseline/p1_resolution.py`、`tests/test_p1_audit.py`。
- 已解决 Bug：无；`DIF-P1-001` 为原始数据审计困难，已按用户批准 policy 解决。
- 遗留困难：`DIF-P10-001` 保持 `OPEN`，不影响 P1 完成结论。
- 阶段门结论：`PASS`
- 完成 commit：本次 P1 完成状态提交。

### P2 完成记录

- 完成日期：2026-08-09
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：canonical XML `nodule >=3 mm` source parser、pylidc physical-nodule clustering、annotation-to-cluster mapping、stable source-derived `nodule_uid`、reader aggregation、per-target valid-reader counts、categorical soft targets/tie flags、computed strict `>3 mm` sensitivity flag、private manifest schema validation 和脱敏 cohort audit。
- Cohort 证据：处理 1,014 个 P1-eligible CT series，生成 2,634 个 physical clusters；primary binary cohort 为 1,073 nodules / 578 patients，`>=3 readers` sensitivity cohort 为 438 nodules；1,560 uncertain clusters 与 1 个明确缺失 `internalStructure` target 的 cluster 未进入 primary binary cohort。reference 2,651 nodules / 875 patients 仅用于 reconciliation，`hard_gate=false`。
- P1 继承结论：4 个不同图像内容 duplicate-plane series 持续排除；14 个超过 4 readers 的 clusters 记录为排除，不任意选择 reader annotation。
- Provenance 与审计：private `artifacts/manifests/nodules.parquet` 和 `annotation_mapping.parquet` 保持 Git ignored；脱敏 `artifacts/audit/p2/summary.json`、`reconciliation.csv`、`exclusions.csv`、`clustering_tolerances.csv` 已提交。每个 eligible series 的 effective clustering tolerance 可审计（范围 `0.10131387882547446–5.0 mm`），不含原始 UID、patient ID 或绝对路径。
- 验收标准与证据：P2-R1–P2-R4 `PASS`；P2 专项 20 项测试及完整 64 项测试均通过；Phase Compliance Reviewer `PASS`，Status Synchronization Reviewer `UPDATED`；用户已明确确认。
- 已解决困难：`DIF-P2-001`；runtime compatibility adapter 仅在缺失时设置 `np.int = int`，未修改 pylidc 源码、SQLite database、依赖锁定或原始数据。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`，不影响 P2 完成结论。
- 明确未纳入内容：P3 consensus mask/ROI、P4 split 及任何模型或训练开发；P3 保持 `NOT_STARTED`。
- 阶段门结论：`PASS`
- P2 实现与审计 commits：`979706a`、`c1c1b95`、`28e46b1`、`d3f3995`；完成状态 commit：本提交。

### 阶段完成记录模板

每个阶段门通过时，在本节追加一条永久记录：

```markdown
### P{N} 完成记录

- 完成日期：YYYY-MM-DD
- 生命周期：COMPLETED
- 健康状态：ON_TRACK
- 已完成内容：
- 明确未纳入内容：
- 验收标准与证据：
- 产物路径：
- 已解决 Bug：
- 遗留困难：
- 阶段门结论：PASS
- 完成 commit：
```

## 10. 状态更新历史

本节只记录阶段切换、阶段阻塞、Bug 模式切换和阶段门结果。

| 日期 | 事件 | 阶段 | 说明 | Commit |
|---|---|---|---|---|
| 2026-08-08 | `PROTOCOL_FROZEN` | 全局 | Baseline-v1 科学需求确认并冻结 | `bdccb98` |
| 2026-08-08 | `PHASE_STARTED` | P0 | 开始工程环境与配置冻结阶段 | 本状态文档初始化提交 |
| 2026-08-09 | `PHASE_AWAITING_APPROVAL` | P0 | 技术验收与双 agent 审查通过，等待用户确认；P1 保持未开始 | 本状态同步提交 |
| 2026-08-09 | `PHASE_COMPLETED` | P0 | 用户确认 P0；阶段封存、交付并完成安全清理，P1 保持未开始 | `a16e3b5` |
| 2026-08-09 | `PHASE_STARTED` | P1 | 用户批准 P1 DICOM/XML 审计实施计划；P1 开始开发，P2 保持未开始 | P1 本地分支 |
| 2026-08-09 | `PHASE_BLOCKED` | P1 | 完整审计发现 5 个 CT series 中的 13 个阻断性 `DUPLICATE_SLICE_PLANE`；审计实现与精简证据已保存为本地 commits，等待用户审阅处理结论，P2 保持未开始 | `2584052`、`3fd5fcb`（未推送） |
| 2026-08-09 | `PHASE_AWAITING_APPROVAL` | P1 | 用户批准的 duplicate-plane policy 已完成并复核：1 个 exact duplicate 仅记录派生 volume selection，4 个不同内容 series 排除，原始 DICOM 未修改；P1 技术验收与阶段级双 agent 审查通过，等待用户确认，P2 保持未开始 | `c70e068`、`7d34cbf`、`fa73a1b`（均未推送） |
| 2026-08-09 | `PHASE_COMPLETED` | P1 | 用户确认 P1；阶段永久记录已写入 4 个不同图像内容 CT series 的排除决定、影响范围和验收证据，P2 保持未开始 | 本次 P1 完成状态提交 |
| 2026-08-09 | `PHASE_STARTED` | P2 | 用户批准 P2 physical nodule cohort 与 stable provenance 实施计划；采用 pylidc 默认 clustering、最小 runtime compatibility adapter 和本地私有 manifest，P3 保持未开始 | P2 本地分支 |
| 2026-08-09 | `PHASE_AWAITING_APPROVAL` | P2 | P2-R1–P2-R4、64 项测试、完整 local cohort audit 和阶段级 Phase Compliance Reviewer 已通过；等待用户确认，P3 保持未开始且不得推送 | `c1c1b95`、`28e46b1`（local, unpushed） |
| 2026-08-09 | `PHASE_COMPLETED` | P2 | 用户确认 P2；永久记录已保存 cohort、provenance、审计与验收证据；P3 保持未开始，随后仅执行合并和推送 | 本次 P2 completion state commit |
