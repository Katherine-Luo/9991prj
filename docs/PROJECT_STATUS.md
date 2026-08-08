---
schema_version: 1
project: LIDC-IDRI Baseline-v1
operating_mode: NORMAL_DEVELOPMENT
reading_scope: CURRENT_AND_NEXT
development_phase: P0
development_phase_status: COMPLETED
maintenance_phase: null
active_bug_ids: []
resume_phase: P0
next_phase: P1
last_updated: 2026-08-09
last_verified_commit: 86cd959
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
| 阶段状态 | `COMPLETED` |
| 维护目标阶段 | 无 |
| 活动 Bug | 无 |
| 当前阻塞项 | 无 |
| 恢复阶段 | `P0` |
| 下一阶段 | `P1 DICOM/XML 审计` |
| 最近更新 | 2026-08-09 |
| 状态依据 commit | `86cd959` |

## 3. 当前阶段：P0 工程环境与配置冻结

### 阶段目标

建立可复现的 Python/PyTorch/MONAI/pylidc 工程环境，完成本地 CPU/MPS 与 Katana CUDA smoke tests，并冻结机器可读的 Baseline-v1 配置及其 SHA-256。

### 已完成

- Baseline-v1 科学需求已确认并冻结。
- Git 仓库已创建并连接 GitHub。
- 冻结需求文档已提交。
- 项目状态文档已建立。
- 仓库级 `AGENTS.md` 开发治理、双 agent 审查和 Git 门禁规则已建立、提交并推送至 `origin/main`。
- Python 3.11 `src` 工程骨架、项目依赖、macOS-arm64 与 Katana-CUDA 环境定义及精确环境锁定文件已完成。
- `configs/baseline_v1.yaml`、只读 resolved config 和 SHA-256 已冻结；配置哈希为 `6a17fd6f3731eb3307cf296fb203e58cf35adb08c3cfd984b136424421fe4a1c`。
- 全局 seed `20260808` 与 `base_seed + fold_index` 派生规则已实现并验证。
- 合成 `64³` ROI 的 CPU、MPS、CUDA DenseNet-121 forward/backward smoke tests 均通过，三种设备使用相同配置哈希。
- Mac 与 Katana 的 `pylidc` import、`pip check` 和 `setuptools 80.10.2` 验证均通过。
- Katana PBS job `8942735.kman.restech.unsw.edu.au` 在 NVIDIA L40S 上以 Exit 0 完成；远程 P0 工作集和剩余空间 gates 均通过。
- 本地完整测试为 `37 passed`；Phase Compliance Reviewer 给出 `PASS`，Status Synchronization Reviewer 已完成状态同步。
- 用户于 2026-08-09 明确批准执行 P0 完成、合并、推送与安全存储清理计划；P0 用户确认门已通过。该记录表示授权已取得，不表示后续交付动作已经执行。

### 正在进行

- P0 已封存完成；正在按已批准计划准备完成状态提交、fast-forward 合并、推送和安全存储清理，P1 尚未开始。

### 尚未完成

- 截至本次完成状态审查，P0 完成状态 commit、`p0-engineering` 到 `main` 的 fast-forward 合并、GitHub 推送和安全存储清理尚未执行；这些是阶段完成后的交付动作，不改变 P0 的 `COMPLETED` 结论。
- P1 实施计划尚未制定或批准，P1 保持 `NOT_STARTED`。

### 验收进度

| P0 验收项 | 状态 | 证据 |
|---|---|---|
| CPU forward/backward smoke test | `PASS` | `artifacts/audit/p0/cpu.json` |
| MPS forward/backward smoke test | `PASS` | `artifacts/audit/p0/mps.json`；记录唯一允许的 `aten::max_pool3d_with_indices` CPU fallback |
| CUDA forward/backward smoke test | `PASS` | `artifacts/audit/p0/cuda.json`；PBS `8942735.kman.restech.unsw.edu.au`，Exit 0，NVIDIA L40S |
| 环境版本已记录 | `PASS` | `artifacts/audit/p0/*-environment.json` 与 `environment/locks/`；两平台 `pylidc` import、`pip check` 通过 |
| 随机 seed 规则已冻结 | `PASS` | base seed `20260808`；fold seed 为 `base_seed + fold_index`；测试通过 |
| Resolved config 与 SHA-256 已冻结 | `PASS` | `configs/baseline_v1.resolved.yaml`、`configs/baseline_v1.sha256`；哈希 `6a17fd6f3731eb3307cf296fb203e58cf35adb08c3cfd984b136424421fe4a1c` |
| 本地自动测试 | `PASS` | 2026-08-09：`37 passed` |
| Katana 存储 gates | `PASS` | `artifacts/audit/p0/katana-storage-preflight.json`；工作集 `7,543,988,928` bytes |
| Phase Compliance Reviewer | `PASS` | 2026-08-09：无冻结需求改动、无 P1 实现、无阻断缺口 |
| Status Synchronization Reviewer | `PASS` | 2026-08-09：状态、代码、测试和审计证据已同步 |

P0 技术验收、双 agent 审查和用户确认均已通过；阶段状态为 `COMPLETED`。

### 未解决困难

- P0 当前无开放困难；`DIF-P0-001` 已在 Katana CUDA 验证通过后标记为 `RESOLVED`。
- `DIF-P10-001` 仍为 `OPEN`，但属于 P10 存储风险，不影响 P0 完成结论。

### 当前证据与产物

- [冻结需求文档](./LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md)
- [仓库开发规则](../AGENTS.md)
- Git commit `bdccb98`：`docs: freeze Baseline-v1 requirements`
- Git commit `d6e37d4`：`docs: add project status tracking`
- Git commit `5388425`：`docs: add repository agent workflow`
- Git commit `d7de8c0`：`docs: align project status approval gates`
- Git commit `8a91863`：`docs: record Katana storage risk`
- Git commit `5f65be3`：`chore: scaffold P0 project environment`
- Git commit `d10d958`：`feat: add deterministic baseline configuration`
- Git commit `bc77067`：`feat: add cross-device DenseNet smoke test`
- Git commit `c1d4393`：`chore: add Katana P0 workflow`
- Git commit `6012941`：`chore: record P0 audit evidence`
- Git commit `1cfac33`：`docs: synchronize P0 project status`
- Git commit `86cd959`：`style: normalize P0 file endings`
- [Baseline-v1 source config](../configs/baseline_v1.yaml)
- [Baseline-v1 resolved config](../configs/baseline_v1.resolved.yaml)
- [Baseline-v1 config SHA-256](../configs/baseline_v1.sha256)
- [P0 audit artifacts](../artifacts/audit/p0/)
- [Platform environment definitions and locks](../environment/)
- [Katana P0 scripts](../scripts/katana/)
- 本地验证命令：`/Users/katherine/.conda/envs/lidc-baseline-v1-p0/bin/python -m pytest -q`；结果 `37 passed`。

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
| P1 | DICOM/XML 审计 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P2 | Physical nodule cohort | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
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
- 完成 commit：本阶段完成状态提交。
- 交付状态：用户已授权；截至完成状态审查，完成状态 commit、fast-forward 合并、推送和安全存储清理尚未执行。

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
| 2026-08-09 | `PHASE_COMPLETED` | P0 | 用户确认 P0；阶段封存为完成，P1 保持未开始 | 本阶段完成状态提交 |
