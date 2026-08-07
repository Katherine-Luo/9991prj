---
schema_version: 1
project: LIDC-IDRI Baseline-v1
operating_mode: NORMAL_DEVELOPMENT
reading_scope: CURRENT_AND_NEXT
development_phase: P0
development_phase_status: IN_PROGRESS
maintenance_phase: null
active_bug_ids: []
resume_phase: P0
next_phase: P1
last_updated: 2026-08-08
last_verified_commit: bdccb98
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
| 当前开发阶段 | `P0 工程环境与配置冻结` |
| 阶段状态 | `IN_PROGRESS` |
| 维护目标阶段 | 无 |
| 活动 Bug | 无 |
| 当前阻塞项 | 无 |
| 恢复阶段 | `P0` |
| 下一阶段 | `P1 DICOM/XML 审计` |
| 最近更新 | 2026-08-08 |
| 状态依据 commit | `bdccb98` |

## 3. 当前阶段：P0 工程环境与配置冻结

### 阶段目标

建立可复现的 Python/PyTorch/MONAI/pylidc 工程环境，完成本地 CPU/MPS 与 Katana CUDA smoke tests，并冻结机器可读的 Baseline-v1 配置及其 SHA-256。

### 已完成

- Baseline-v1 科学需求已确认并冻结。
- Git 仓库已创建并连接 GitHub。
- 冻结需求文档已提交。
- 项目状态文档已建立。

### 正在进行

- 建立 P0 开发状态和后续阶段维护规则。

### 尚未完成

- 创建 Python 项目骨架与环境文件。
- 锁定 Python、PyTorch、MONAI、pylidc、pydicom、pandas/pyarrow 和 scikit-learn/scipy 版本。
- 完成 CPU、MPS 和 CUDA forward/backward smoke tests。
- 固定全局 seed 与 fold seed 派生规则。
- 创建 `configs/baseline_v1.yaml`。
- 生成只读 resolved config 和 SHA-256。
- 保存环境版本和 CUDA 信息。

### 验收进度

| P0 验收项 | 状态 | 证据 |
|---|---|---|
| CPU forward/backward smoke test | `PENDING` | — |
| MPS forward/backward smoke test | `PENDING` | — |
| CUDA forward/backward smoke test | `PENDING` | — |
| 环境版本已记录 | `PENDING` | — |
| 随机 seed 规则已冻结 | `PENDING` | — |
| Resolved config 与 SHA-256 已冻结 | `PENDING` | — |

P0 阶段门当前未通过。

### 未解决困难

- `DIF-P0-001`：Katana CUDA 环境和依赖版本尚未验证。它不阻塞本地工程初始化，但阻止 P0 阶段门通过。

### 当前证据与产物

- [冻结需求文档](./LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md)
- Git commit `bdccb98`：`docs: freeze Baseline-v1 requirements`

## 4. 下一阶段：P1 DICOM/XML 审计

### 阶段目标

确认 canonical XML source，盘点 CT/DX/CR/CXR 数据，建立 XML–DICOM UID 映射，并验证 CT series 的几何完整性和确定性切片排序。

### 进入条件

- P0 全部验收项通过。
- Baseline-v1 resolved config 和 SHA-256 已冻结。
- 本地审计环境可以读取 DICOM headers 和 canonical XML。

### 第一批任务

- 统计本地 patient directories、DICOM modalities、CT series 和 XML 类型。
- 确认 `LIDC-XML-only` 为唯一 canonical annotation source。
- 建立 Study/Series/SOP UID 映射。
- 检查 orientation、position、spacing、重复切片和异常 slice gaps。
- 生成可审计的异常清单和 Phase 1 报告。

### 已知风险

- Canonical XML 中包含 CXR XML，必须与 CT annotations 分离。
- DICOM 下载目录内的 XML 只能用于交叉核对，不能与 canonical XML 合并生成重复 annotation。
- 切片必须根据空间坐标投影排序，不能只依赖文件名或 InstanceNumber。

<!-- NORMAL_READING_END -->

---

以下为完整历史与维护区。正常开发模式无需每次通读；`BUG_MAINTENANCE` 模式必须从此处继续阅读到文件末尾。

## 5. 状态模型与更新规则

### 阶段生命周期

| 状态 | 含义 |
|---|---|
| `NOT_STARTED` | 尚未进入该阶段 |
| `IN_PROGRESS` | 正在开发或重新验收 |
| `BLOCKED` | 阶段无法继续，且已有明确阻塞条件 |
| `COMPLETED` | 全部阶段门验收已通过并保存证据 |

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
3. 如果阶段门通过，将阶段标记为 `COMPLETED`，追加永久阶段记录，并将下一阶段切换为 `IN_PROGRESS`。
4. 状态更新与对应代码或修复放在同一个 Git commit；不复制普通 Git commit 日志。

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
4. 清除 `maintenance_phase` 和 `active_bug_ids`，恢复 `NORMAL_DEVELOPMENT / CURRENT_AND_NEXT`，回到 `resume_phase`。

### 未解决困难

困难不得因阶段切换而删除。其状态只能在以下值间变化：

- `OPEN`
- `MITIGATED`
- `RESOLVED`

## 6. 完整阶段总表

| 阶段 | 名称 | 生命周期 | 健康状态 | 阶段门 | 开放 Bug | 开放困难 |
|---|---|---|---|---|---:|---:|
| P0 | 工程环境与配置冻结 | `IN_PROGRESS` | `ON_TRACK` | 未通过 | 0 | 1 |
| P1 | DICOM/XML 审计 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P2 | Physical nodule cohort | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P3 | Consensus mask 与 ROI | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P4 | Patient-level split 与共享初始化 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P5 | Black-box DenseNet | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P6 | Standard CBM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P7 | Mixed-type CEM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P8 | CBM + GAM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P9 | 统一评估 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P10 | Katana 正式实验与报告 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |

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

### DIF-P0-001：Katana CUDA 环境尚未验证

- 状态：`OPEN`
- 所属阶段：P0
- 首次记录：2026-08-08
- 影响：无法验证正式训练环境的 CUDA forward/backward smoke test，因此 P0 阶段门不能通过。
- 当前结论：不阻塞本地 CPU/MPS 工程初始化。
- 下一步：建立项目环境后，在 Katana GPU node 上记录 CUDA、驱动、PyTorch 和 MONAI 版本并运行 smoke test。
- 解除条件：Katana smoke test 通过，结果和环境版本写入 P0 验收证据。
- 关联 Bug：无。

## 9. 阶段永久记录

### P0 当前未封存记录

- 开始日期：2026-08-08
- 当前状态：`IN_PROGRESS`
- 已完成：需求冻结、仓库创建与连接、冻结需求文档提交、状态文档建立。
- 未完成：项目骨架、依赖锁定、三种设备 smoke tests、机器可读配置及哈希。
- 当前验收结果：未通过。
- 开放困难：`DIF-P0-001`。
- 开放 Bug：无。
- 阶段完成日期：—
- 阶段完成 commit：—

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
