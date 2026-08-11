---
schema_version: 2
project: LIDC-IDRI Baseline-v2
active_protocol: Baseline-v2
active_requirements: docs/LIDC_IDRI_BASELINE_V2_REQUIREMENTS.md
active_config: configs/baseline_v2.yaml
supersedes_protocol: Baseline-v1
protocol_transition: V2M
operating_mode: NORMAL_DEVELOPMENT
reading_scope: CURRENT_AND_NEXT
development_phase: P6
development_phase_status: IN_PROGRESS
maintenance_phase: null
active_bug_ids: []
resume_phase: P6
next_phase: P7
last_updated: 2026-08-11
last_verified_commit: 5aedec3
---

# LIDC-IDRI Baseline-v2 项目状态

本文件是项目开发状态的唯一事实来源。当前所有开发只依据已批准并冻结的 [Baseline-v2 需求文档](./LIDC_IDRI_BASELINE_V2_REQUIREMENTS.md)和 `configs/baseline_v2.yaml`；Baseline-v1 已被取代，仅保留用于历史审计，不得作为后续实现依据。V2M、P3、P4 与 P5 均已完成、获用户确认并推送；P5 最终交付 anchor 为 `c392c04`。P6 最终实施计划已获用户明确批准，当前为唯一允许开发的阶段；P7 保持 `NOT_STARTED`。

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
| Active protocol | `Baseline-v2` |
| Historical protocol | `Baseline-v1`（`SUPERSEDED`，audit-only） |
| 当前开发阶段 | `P6 Sequential Standard CBM Regression` |
| 阶段状态 | `IN_PROGRESS / ON_TRACK` |
| 维护目标阶段 | 无 |
| 活动 Bug | 无 |
| 当前阻塞项 | 无。P6 execution supplement与core primitives已完成本地验证；完整sequential lifecycle、cache persistence、test-once transaction、Stage A、正式训练和远程作业尚未完成。 |
| 恢复阶段 | `P6` |
| 下一阶段 | `P7 Mixed-type CEM`（保持 `NOT_STARTED / NOT_APPLICABLE`） |
| 最近更新 | 2026-08-11 |
| 状态依据 | P6 启动状态、execution supplement和core primitives原子commits依次为 `881df05`、`c3224f4`、`5aedec3`，均仅存在于本地P6分支、尚未推送。Core primitives实现八个独立linear heads、activated canonical 16D vector、八组等权concept loss、sample-weighted aggregation、deterministic head initialization、partition records/dataset、predictor freeze/BatchNorm hash、predicted-cache feature guard与两种量纲的贡献重建。Direct tests `16 passed`、完整测试 `189 passed`，最终Phase Compliance Reviewer为`PASS`。完整sequential lifecycle、cache persistence、checkpoint/resume、test-once transaction、Stage A和正式训练尚未完成。P6保持`IN_PROGRESS / ON_TRACK`，P7保持`NOT_STARTED`。 |

## 3. 当前阶段：P6 Sequential Standard CBM Regression

### 阶段目标

在 P4 固定 patient-level splits、共享 DenseNet-121 encoder initialization 和 P5 已验证的 common H200 warn-only execution profile上，实现两阶段 sequential Standard CBM：先训练八组 concept predictor，再使用 frozen activated predicted concepts 训练 unconstrained linear malignancy regression head。

### 已完成

- P5 已完成、确认并推送；P6 的数据、ROI、splits、shared encoder initialization、H200 execution profile和训练公平性规则均可复用。
- 用户已批准 P6 最终实施计划与四项实现澄清。
- 已从已交付 `main` 创建本地分支 `p6-standard-cbm`。
- P6 execution supplement、deterministic resolved config 与 SHA-256 已生成；resolved SHA-256 为 `792f544aef33d30f122054ba40bdf8f185cea71e516614545ba3f85879ed3bc3`。
- Execution supplement 已固定 independent linear concept heads、activated canonical 16D task input、train/validation-only pre-task cache、frozen predicted concepts、test-after-task-best transaction、deterministic head seed domains、两阶段各 80 epochs及 H200 common profile引用。
- 配置专项测试 `4 passed`、完整测试 `177 passed`，Phase Compliance Reviewer 为 `PASS`；冻结 V1/V2 requirements/config 未修改。
- Core primitives commit `5aedec3`已实现并验证八个独立linear concept heads、sigmoid/softmax activated canonical 16D vector、八组等权loss、partial-batch sample-weighted aggregation、fold-specific deterministic head initialization、manifest/split/ROI partition records与dataset、predictor freeze/BatchNorm state hashing、predicted-cache schema guard以及normalized/rating-scale contribution reconstruction。
- Core primitives direct tests `16 passed`、完整测试 `189 passed`，最终Phase Compliance Reviewer为`PASS`；冻结requirements/config无diff。

### 正在进行

- 准备下一原子批次的完整sequential lifecycle、cache persistence、checkpoint/resume、test-once transaction与Katana接口；尚未执行Stage A或任何正式训练。

### 尚未完成

- Concept/task两阶段的完整训练、checkpoint、scheduler与resume lifecycle。
- Leakage-safe train/validation concept cache持久化、provenance验证和test-once transaction。
- H200 Stage A、五折formal runs、OOF、脱敏audit及阶段门。

### 验收进度

| P6 验收项 | 状态 | 证据 |
|---|---|---|
| P6 execution supplement、resolved config与hash | `PASS` | SHA-256 `792f544aef33d30f122054ba40bdf8f185cea71e516614545ba3f85879ed3bc3`；专项 `4 passed`、完整 `177 passed`；Phase Compliance Reviewer `PASS` |
| P6-R1 concept predictor与八组等权loss | `IN_PROGRESS` | Core predictor/loss/data primitives与direct `16 passed`已验证；完整训练lifecycle和正式fold证据待完成 |
| P6-R2 sequential frozen-prediction task training | `NOT_STARTED` | 待实现 |
| P6-R3 normalized/rating contribution reconstruction | `IN_PROGRESS` | Unit-level exact reconstruction≤`1e-6`已验证；正式fold predictions重建证据待完成 |
| H200 Stage A、五折OOF与双agent阶段门 | `NOT_STARTED` | 待执行 |

### 未解决困难

- 当前无 P6 特有开放困难；`DIF-P10-001`继续开放但不阻止P6。

## 4. 下一阶段：P7 Mixed-type CEM

### 阶段目标

仅在 P6 完成、用户确认并推送后另行制定；当前不实现或详细规划。

### 进入条件

- P6 必须通过全部技术验收、双agent审查、用户确认、合并与GitHub推送。

### 第一批任务

- 尚未制定或批准；P7保持`NOT_STARTED`。

<!-- NORMAL_READING_END -->

## 5. P5 历史开发快照

### 阶段目标

在 P4 固定 split 与共享 encoder initialization 上实现并完成 Black-box 3D DenseNet-121 连续 malignancy score regression。P5 分为 Stage A Fold 0 正式门和用户中间确认后的 Stage B Folds 1–4；本阶段只训练 Black-box，不进入 Standard CBM。

### 已完成

- P4 已完成、确认、合并并推送；2,633 个 private ROI、五折 patient-grouped split、train-only statistics 和每折共享 DenseNet-121 encoder initialization 是 P5 固定输入。
- P4 最终状态已由 `960e366` 交付，P4 L40S job `8962963.kman.restech.unsw.edu.au` 为 `Exit_status=0`，remote workset 与 CUDA forward smoke 可复用。
- 用户已批准 P5 两阶段计划：先完成 Fold 0 formal gate 并等待中间确认，再以完全相同 execution config 执行 folds 1–4；P5 最终必须产生 2,633 个 OOF predictions。
- 用户已固定 reference-aligned common policy：Adam、`lr=1e-4`、80 epochs、batch 16、4 个 validation bad epochs 后 LR 乘 `0.9`，以及仅训练集使用的 rotation/flips/z-order reversal。
- 用户已固定 Fold-0 前实现澄清：5D rotation 使用 `mode=bilinear`、`padding_mode=zeros`、`align_corners=false`；`drop_last=false`；Black-box head 使用 fold-specific domain-separated deterministic seed/hash；所有参考论文未精确报告的细节均标记为 Baseline-v2 project pre-registered choices。
- 已创建本地分支 `p5-blackbox-regression`；冻结 V1/V2 requirements/config/resolved/hash 无 diff。
- 原始 `configs/experiments/baseline_v2_reference_training.yaml`、resolved config 与 hash `afadd6a6944bb7e7886a9dcb68781a9389e4b3afbea402dd23418494c30b2327` 已由 `68cc73e` 提交并随 P5 交付推送；其 reference-aligned optimizer/scheduler/batching/augmentation choices 继续有效，但 L40S execution profile 现仅保留为历史，不能驱动正式 P5–P8 runs。
- 已实现用户批准的 H200 execution/hardware profile amendment：H200 是 P5–P8 统一正式训练 GPU，不改变 `configs/baseline_v2.yaml` 或科学协议。新的 `baseline_v2_reference_training_h200.yaml`、resolved config 和 SHA-256 固定原有训练策略与 FP32/no-AMP/no-BF16/no-TF32 约束，config hash 为 `08df87e4be5f07985d9dd3619b471ad322ec23a4b98b5032ee05ed58b1918281`；相应 config、代码、PBS scripts 和 tests 已由 `c5ee485` 提交并随 P5 交付推送，Phase Compliance Reviewer 为 `PASS`。旧 strict profile 仅保留为历史；active warn-only H200 profile 已通过 Stage A 与五折正式执行。
- `BUG-P5-001` 已由 commit `11658ab` 修复并随 P5 交付推送：独立的 `baseline_v2_reference_training_h200_warn_only` execution/reproducibility profile 继续固定 H200 为 P5–P8 统一正式训练 GPU，并保持 FP32、禁用 AMP/BF16/TF32、batch 16、optimizer、scheduler、augmentation、数据和科学 config 不变。profile 明确记录 `torch.use_deterministic_algorithms(True, warn_only=True)`：不支持的 CUDA deterministic operations 发出 warning 而不阻断 backward，且不宣称 CUDA training bitwise-identical。resolved SHA-256 为 `66c925a7b43bf9fa312ceb850b43746a34d1808888667c39392eaef9e47495bb`；相关代码、tests、PBS 和 transfer-manifest 均已由该 commit 封存。完整测试为 `170 passed`，P5 direct tests 为 `35 passed`；KDM sync、remote integrity verify、H200 Stage A、Bug 修复 Phase Compliance Reviewer 与用户继续 P5 的确认均已通过。
- 已实现 `src/lidc_baseline/p5_blackbox.py` 及 direct tests，覆盖 frozen execution-config enforcement、P4 shared encoder hash 验证、fold-specific deterministic unconstrained linear head、manifest/split/ROI data loading、train-only deterministic augmentation、Adam 与 validation-MSE scheduler、80-epoch training/checkpoint selection、atomic checkpoint/history artifacts、完整 RNG/optimizer/scheduler resume、single-writer fold lifecycle、one-time test transaction/recovery、prediction provenance、unclipped regression metrics、verify、overfit-check 和 active-profile H200 batch-16 preflight interfaces。
- Core direct tests 曾为 `26 passed`、完整测试曾为 `161 passed`；warn-only 修复后，P5 direct tests 为 `35 passed`、完整测试为 `170 passed`。冻结协议检查和修复批次 Phase Compliance Reviewer 均为 `PASS`。上一轮 interim `FAIL` 的五项阻断发现已全部修复并由最终合规复核验证；它们属于提交前审查缺口，未造成已交付阶段或正式结果失效，因此不登记为 Bug。Core batch `64f01c7` 与 warn-only remediation `11658ab` 均已随 P5 交付推送。H200 Stage A、Formal Fold 0 scientific execution 及 existing-artifact final verify 均已通过。
- 已实现 `src/lidc_baseline/p5_katana.py`、`sync_p5_stage_a.sh`、`p5_stage_a.pbs`、`p5_fold.pbs` 和 direct tests。H200 amendment 将 Stage A/formal PBS GPU request 与 execution config 均切换至 H200；formal-fold PBS 继续支持 resume/one-time test/verify，并在未获 Stage B 批准时阻止 folds 1–4。
- H200 private transfer manifest 已验证并完成 KDM sync：P4 immutable base 保持不变，H200 P5 delta 为 7 files / `92,118` bytes；transfer manifest SHA-256 为 `d15f5f95f67983f4e51e7a1a0275611b7d786f8eee62e3c171644a78510e83a0`，remote integrity verify 为 `PASS`。旧 L40S job `8964315.kman.restech.unsw.edu.au` 已在 `Q` 状态取消且未运行；新 H200 job `8964634.kman.restech.unsw.edu.au` 后续已启动并以 Exit 1 失败，详见 `BUG-P5-001`。
- 已实现并完成 P5 aggregate audit：`p5_audit.py` 验证 private runs、one-time tests、profile-bound formal provenance、CUDA H200 runtime 和纯 FP32 invariants（AMP/BF16/CUDA matmul TF32/cuDNN TF32 均关闭），并以兼容真实 Stage A outputs 的独立 provenance schema 验证 overfit/preflight 证据。Audit implementation 与 tests 由 `dff1356`、H200 amendment 由 `c5ee485`、五折 OOF audit 由 `a81d06b` 封存，六个脱敏 tracked JSON 由 `0359d61` 封存；相关 commits 均已随 P5 推送。
- H200 job `8964634` 已实际获得 `k205` NVIDIA H200 并开始执行；queue wait `eligible_time=00:04:16`，remote integrity 再次为 `PASS`。Overfit backward 随后因 `avg_pool3d_backward_cuda` 缺少 deterministic implementation 而失败，job `Exit_status=1`；因此已确认硬件分配、数据完整性和远程工作集不是本次失败原因。重跑 job `8965003` 随后在 `k220` 的 NVIDIA H200 GPU 2 成功完成（Exit 0）；它通过 8-sample/40-step overfit 和 true-batch-16 forward/MSE/backward/Adam preflight，未启动 formal Fold 0。
- 用户在 Stage A 通过后明确批准继续 P5；唯一获授权的 formal Fold 0 job `8965243.kman.restech.unsw.edu.au` 已在 `k205` H200 GPU 7 完成 80 epochs、minimum-validation-MSE best checkpoint 固定和一次性 479-sample test。训练于 10:37:39 AEST 启动，GPU 计算于 11:02:16 结束，PBS 于 11:02:22 以 Exit 1 结束；失败仅来自 test 后 final verify 的 float round-trip 零容差误报。Private run directory 为约 `260 MB`，所有 checkpoint、history、predictions、plots、metrics、runtime 和 one-time-test evidence 均已保留。
- Fold 0 完整 scientific output 已保存：best checkpoint SHA-256 `e6db39216c3a0253dddee4761d8d99fc9f4550ba58c3414b8b6eaff6b25fb810`；479-sample normalized MAE/RMSE `0.12464988572819828/0.1609135068084677`，original-scale MAE/RMSE `0.4985995429127931/0.6436540272338708`，Pearson/Spearman `0.7273738908734001/0.6499290631630589`；prediction range `[-0.011651983484625816, 0.971398651599884]`，below-0 与 below-1-original rates 均为 `0.0020876826722338203`，above-1 与 above-5-original rates 均为 `0.0`。运行固定 `FP32=true`、AMP/BF16/TF32 全部关闭、`torch_use_deterministic_algorithms=true`、`warn_only=true`；PyTorch/MONAI/CUDA 为 `2.5.1+cu121/1.4.0/12.1`。
- `BUG-P5-002` 已由 verifier-only commit `2eaa273` 修复：JSON/CSV best-objective consistency 使用 `math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)`；tiny serialization round-trip difference 被接受，真实 mismatch 仍被拒绝。P5 direct/full tests 为 `33 passed` / `172 passed`，Phase Compliance Reviewer 为 `PASS`。新代码经 KDM 同步后 remote integrity 为 `PASS`；只对原有 Fold 0 artifacts 重新执行 final verifier，返回 `PASS`（80 epochs、best epoch index `14`、validation MSE `0.0199759813899636`、test-once 479 samples、每 epoch 1,882 train samples）。未重训、未重新执行 test，checkpoint、history、predictions 和 metrics 的 hashes/mtimes 均保持不变。用户已批准 Fold 0 scientific execution 及该 verifier-only repair；Bug 关闭后 P5 恢复 `NORMAL_DEVELOPMENT / IN_PROGRESS / ON_TRACK`。
- 用户已明确批准 Stage B，folds 1–4 已使用与 Fold 0 相同的 frozen H200 warn-only profile 完成：jobs `8965994`、`8965995`、`8965996`、`8965997` 均运行 80 epochs，按 minimum validation MSE 固定 best checkpoint 后各执行一次 test，final verifier 均为 `PASS` 且 `Exit_status=0`。Best epoch indexes 为 `[19,10,15,38]`，对应 validation MSE 为 `[0.0247182929771812,0.0238741965997411,0.0228595164578849,0.0193156898568515]`；连同 Fold 0 的五折 best epoch indexes 为 `[14,19,10,15,38]`。
- CPU OOF job `8966614.kman.restech.unsw.edu.au` 已在 `k189` 以 `Exit_status=0` 完成。五折 OOF 精确覆盖 2,633 nodules / 868 patients，fold test counts 为 `[479,502,539,549,564]`，每个 nodule 恰有一次 test prediction，patient leakage 为 0。Pooled OOF original-scale MAE/RMSE 为 `0.50062578502153/0.6421887532919749`，normalized MAE/RMSE 为 `0.1251564462553825/0.16054718832299372`，Pearson/Spearman 为 `0.7157050987783329/0.634502488281551`；prediction min/max 为 `-0.12787137925624847/1.0300079584121704`，below-0/above-1 rates 为 `0.014812001519179644/0.001139384732244588`。
- Private `oof_predictions.parquet` 仅保留在 Katana，未复制或提交至本地仓库；其 SHA-256 为 `6f7e8b840638cfcce3427a1a1e63155860f1067ac6d09f10e7c43aa74a2763e8`。Tracked evidence 仅为六个脱敏 JSON：`fold_0.json`–`fold_4.json` 和 `summary.json`。五折 private run storage 合计 `1,360,388,058` bytes。
- OOF/audit implementation、CPU-only PBS 与 transfer whitelist 已由 commit `a81d06b`（`feat: add P5 five-fold OOF audit`）封存，六个脱敏 aggregate audit JSON 已由 commit `0359d61`（`data: add P5 deidentified OOF audit`）封存。最新 P5 delta 为 9 files / `113,468` bytes，manifest content SHA-256 为 `5523ce78167e4c28f0ba4e1debdda49ab71be9856af6a0121e937f371d375a5d`，remote integrity 为 `PASS`。P5 audit/Katana direct tests 为 `8 passed`，完整 tests 为 `173 passed`；阶段级 Phase Compliance Reviewer 为 `PASS`，冻结 V1/V2 requirements/config 无 diff。
- Approval-gate status `6e07bb5` 与 completion status `147f8f0` 已封存；用户于 2026-08-11 明确确认 P5。P5 已 fast-forward 合并至 `main` 并推送 GitHub；合并后完整 tests 为 `173 passed`，冻结检查与 post-delivery Phase Compliance Reviewer 均为 `PASS`，本地 `main`、`HEAD` 与 `origin/main` SHA 一致。P5 生命周期为 `COMPLETED`、健康状态为 `ON_TRACK`，P6 继续保持 `NOT_STARTED`。

### 正在进行

- P5 开发、阶段确认与 Git 交付均已完成；当前仅同步 post-delivery 状态，不开展新的 P5 功能或 P6 开发。

### 尚未完成

- P6 的实施计划尚未制定或批准，保持 `NOT_STARTED`。

### 验收进度

| P5 验收项 | 状态 | 证据 |
|---|---|---|
| H200 execution/hardware profile amendment | `PASS` | H200 为 P5–P8 统一正式训练 GPU且不改变科学协议；新 config hash `08df87e4be5f07985d9dd3619b471ad322ec23a4b98b5032ee05ed58b1918281`。旧 L40S profile 保留为历史并被 formal validation 拒绝；本批次由 `c5ee485` 提交并已随 P5 推送，KDM/remote verify 与 Phase Compliance Reviewer 均为 `PASS` |
| H200 warn-only execution/reproducibility remediation | `PASS` | Commit `11658ab` 封存独立 profile、代码、PBS、provenance、tests 和 transfer manifest，并已随 P5 推送；resolved SHA-256 为 `66c925a7b43bf9fa312ceb850b43746a34d1808888667c39392eaef9e47495bb`，manifest 为 7 files / `94,596` bytes、SHA-256 `fa1acbe5...c3c36b`。完整测试 `170 passed`、P5 direct tests `35 passed`、Bug 修复 Phase Compliance Reviewer `PASS`；KDM sync、remote integrity 和 job `8965003` Stage A 均已通过，用户已明确批准继续 P5 |
| P5 core model/data/augmentation/scheduler/checkpoint/resume/test transaction interfaces | `PASS` | `p5_blackbox.py` 与 direct tests 已实现并由 `64f01c7` 提交；verifier-only float consistency fix 为 `2eaa273`。修复后 P5 direct/full tests 为 `33 passed` / `172 passed`，冻结检查与 Phase Compliance Reviewer 均为 `PASS`；相关 commits 已随 P5 推送 |
| H200 Katana Stage A transfer/PBS interfaces | `PASS` | 旧 strict-profile delta（7 files / `92,118` bytes，manifest `d15f5f95...e83a0`）的 KDM sync/remote verify 为 `PASS`，但该 profile 已不能驱动修复后的 Stage A。新 warn-only delta（manifest `fa1acbe5...c3c36b`）已完成 KDM sync 与 remote re-verify `PASS`；job `8965003` 显式请求 H200，并已在 `k220` GPU 2 以 Exit 0 完成 |
| P5 aggregate audit、OOF 与脱敏 evidence | `PASS` | CPU OOF job `8966614` 在 `k189` 以 Exit 0 完成；OOF 2,633 nodules / 868 patients、fold counts `[479,502,539,549,564]`、0 patient leakage。Private OOF 仅在 Katana，SHA-256 `6f7e8b840638cfcce3427a1a1e63155860f1067ac6d09f10e7c43aa74a2763e8`；tracked evidence 仅为五个 fold JSON 与一个 summary JSON，均已通过脱敏检查 |
| Fold 0 train-only overfit 与 H200 batch-16 preflight | `PASS` | Job `8964634` 在 `k205` H200 上因旧 strict profile Exit 1；warn-only job `8965003` 已在 `k220` GPU 2 以 Exit 0 完成。8 samples/40 steps overfit MSE 从 `0.1297724843` 降至 `0.0642339364`；true batch 16 forward/MSE/backward/Adam 均通过；peak reserved `2,860,515,328 / 150,393,585,664 bytes = 1.9020%`，低于 85% 门槛。运行发出预期的 AvgPool3d 与 MaxPool3d warn-only warnings |
| Fold 0 formal 80 epochs、best checkpoint、一次性 test 与 final verify | `PASS` | job `8965243.kman.restech.unsw.edu.au` 在 `k205` H200 GPU 7 完成 80 epochs；每 epoch 使用全部 1,882 train nodules。Best epoch index `14`、validation MSE `0.01997598138996362`；best checkpoint固定后 test exactly once 共479 samples。Test original-scale MAE/RMSE=`0.4985995429127931/0.6436540272338708`，normalized MAE/RMSE=`0.12464988572819828/0.1609135068084677`，Pearson/Spearman=`0.7273738908734001/0.6499290631630589`，prediction min/max=`-0.011651983484625816/0.971398651599884`。Verifier-only fix `2eaa273` 后只对既有 artifacts 运行 remote final verify，返回 `PASS`；未重训或重复 test，private artifact hashes/mtimes 不变 |
| Folds 1–4、五折 OOF 与 checkpoint/test invariants | `PASS` | Stage B jobs `8965994`–`8965997` 均 80 epochs、test exactly once、final verifier `PASS`、Exit 0。五折 best epochs `[14,19,10,15,38]`，validation MSE `[0.0199759813899636,0.0247182929771812,0.0238741965997411,0.0228595164578849,0.0193156898568515]`；OOF 2,633/2,633 且 0 leakage。Pooled original-scale MAE/RMSE=`0.5006257850/0.6421887533`，Pearson/Spearman=`0.7157050988/0.6345024883` |
| 冻结协议保护、测试、双 agent 审查与阶段治理 | `PASS_DELIVERED` | 冻结 V1/V2 requirements/config 无 diff；P5 audit/Katana direct tests `8 passed`、合并后完整 tests `173 passed`。Phase Compliance Reviewer、Status Synchronization Reviewer 与 post-delivery Phase Compliance Reviewer 均通过，用户于 2026-08-11 明确确认。Completion commit `147f8f0` 与 post-delivery 状态同步 commit `c392c04` 均已推送，本地 `main`、`HEAD` 与 `origin/main` 三方均为 `c392c04c556a563c4b1fefd6ae69c3735c742083`；P6 未开始 |

### 未解决困难

- `DIF-P10-001` 继续为 `OPEN`，当前不阻止 P5；P5 必须记录 checkpoints、history、predictions 与 runtime 的实际 storage，供 P10 总工作集估算使用。

### P6 启动前历史快照

### 阶段目标

在与 P5 相同的 split、共享 encoder 初始化和 common execution policy 下实现 sequential Standard CBM。

### 进入条件

- P5 必须完成全部五折、技术验收、最终用户确认、合并和 GitHub 推送。
- P6 必须另行制定实施计划并获得用户明确批准。

### 第一批任务

- 尚未制定或批准；P6 保持 `NOT_STARTED`。

<!-- P5_HISTORICAL_SNAPSHOT_END -->

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
| V2M | Baseline-v2 Protocol Migration | `COMPLETED` | `ON_TRACK` | V2M-R1–V2M-R5、86 项测试、双 agent 审查和用户确认均为 `PASS`；已推送 | 0 | 0 |
| P3 | Consensus mask 与 ROI | `COMPLETED` | `ON_TRACK` | P3-R1–P3-R3、冻结协议保护、full 2,633 ROI verify、32 项 P3 tests、118 项完整 tests、aggregate audit、阶段级双 agent 审查和用户最终确认均为 `PASS`；已由 `dc8c356` 合并并推送，P3 完成时 P4 尚未开始 | 0 | 0 |
| P4 | Patient-level split 与共享初始化 | `COMPLETED` | `ON_TRACK` | P4-R1–P4-R3、实际 KDM sync、L40S CUDA smoke、tracked audit、P4 `17 passed`、合并前后完整 `135 passed`、阶段级双 agent 审查、completion-sealing/post-delivery Phase Compliance Reviewers 和用户确认均为 `PASS`；evidence、approval-gate、delivery anchors 分别为 `9d24035`、`e0634e7`、`ec7bd8e`，已合并并推送，P5 未开始 | 0 | 0 |
| P5 | Black-box DenseNet regression | `COMPLETED` | `ON_TRACK` | 五折 80 epochs、minimum-validation-MSE checkpoints、test exactly once、final verifies、2,633/868 OOF、0 leakage、tracked audit、direct `8 passed`、合并后完整 `173 passed`、阶段级与 post-delivery 审查及用户 2026-08-11 确认均为 `PASS`；completion `147f8f0` 与 post-delivery sync `c392c04` 均已推送，P6 未开始 | 0 | 0 |
| P6 | Standard CBM | `IN_PROGRESS` | `ON_TRACK` | Execution supplement与core primitives已验证；core direct `16 passed`、完整 `189 passed`、Phase Compliance Reviewer `PASS`。完整sequential lifecycle、Stage A及正式训练尚未完成 | 0 | 0 |
| P7 | Mixed-type CEM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P8 | CBM + GAM | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P9 | 统一评估 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 0 |
| P10 | Katana 正式实验与报告 | `NOT_STARTED` | `NOT_APPLICABLE` | 未执行 | 0 | 1 |

## 7. Bug 登记表

### 活动 Bug

当前无活动 Bug。P5 为 `COMPLETED / ON_TRACK`；五折 formal runs、one-time tests、final verifies、OOF reconciliation、测试、双 agent 审查、用户确认、Git 交付与 post-delivery 状态同步均为 `PASS`。P6 当前为 `NORMAL_DEVELOPMENT / IN_PROGRESS / ON_TRACK`，P7–P8 保持 `NOT_STARTED`。`BUG-P5-002`、`BUG-P5-001`、`BUG-P3-001` 与 `BUG-P3-002` 均已解决。

### Bug 状态

`OPEN` → `INVESTIGATING` → `FIXING` → `VERIFYING` → `RESOLVED`

### BUG-P5-002：Verifier 对 CSV/JSON float round-trip 使用零容差比较

- 状态：`RESOLVED`
- 严重度：`MEDIUM`
- 发现日期：2026-08-11
- 影响阶段：P5
- 影响验收标准：是；P5-R2 的 checkpoint 选择和 test-only-after-selection 科学执行均已满足，但 final automated verify 误报失败，阻止 Fold 0 技术门与 tracked aggregate audit 完成。
- 恢复阶段：P5
- 受影响下游阶段：无已启动阶段；修复验证后不再影响下游。此后 folds 1–4 已获批准并完成，P6 继续保持 `NOT_STARTED`。
- 现象：Formal Fold 0 job `8965243.kman.restech.unsw.edu.au` 已完成 80 epochs，固定 epoch index `14` 的 minimum-validation-MSE checkpoint，并对 479 个 test samples 执行且仅执行一次 evaluation。随后 final `verify` 将 JSON 中的 best validation MSE `0.01997598138996362` 与 `history.csv` 经 pandas 读取后的 `0.0199759813899636` 直接比较，约 `2.08e-17` 的表示差触发 `ValueError: P5_BEST_OBJECTIVE_MISMATCH`，PBS 最终 `Exit_status=1`。
- 复现方式：对 job `8965243` 已有 private artifacts 运行 P5 Fold 0 `verify`；无需加载训练数据或重新评估 test，即可在 best-objective consistency check 复现。
- 根因：相同 validation MSE 在 JSON 和 CSV 两种序列化路径之间发生正常的 IEEE-754 decimal round-trip 表示差；verifier 使用零容差精确相等而不是明确的 float comparison policy，因此产生 false failure。该差异不改变 minimum epoch、best checkpoint、prediction 或 metrics。
- 修复：用户批准的 verifier-only commit `2eaa273` 将跨 JSON/CSV best-objective consistency check 改为 `math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)`；新增 tiny serialization-difference positive test 与真实 objective mismatch negative test。未修改 checkpoint、history、predictions、metrics、frozen training profile 或科学协议。
- 验证命令与结果：P5 direct tests `33 passed`，完整 tests `172 passed`，Phase Compliance Reviewer `PASS`。KDM delta 为 7 files / `95,099` bytes，transfer-manifest content SHA-256 `75e14c9e1fcc3cbf2bd718dbe3e7ee8415adcf7875e3861bcd6c2dc1160db32d`，manifest file SHA-256 `576a7e1e52b6c7d8a7b0b51f7886a147b6e72f75a14354b9e780b1ff81ca19e8`，remote integrity `PASS`。只对 job `8965243` 的既有 Fold 0 artifacts 重新运行 final verifier，返回 `PASS`：80 epochs、best epoch index `14`、validation MSE `0.0199759813899636`、test-once 479 samples、每 epoch 1,882 train samples。未调用 train 或 evaluate-test；best checkpoint SHA-256 `e6db39216c3a0253dddee4761d8d99fc9f4550ba58c3414b8b6eaff6b25fb810`、prediction SHA-256 `af68e6f9821e207dbd2e6fc9f7391ac98e4553256a7e02cad53ee8ba37b4ad74`、metrics SHA-256 `7b8a91342f939e0c26a202136e08fceddddbd2c39a23033af937fe8f8427ff70` 及相关 mtimes 均保持不变。
- 未解决事项：无。Fold 0 gate、folds 1–4 与五折 OOF 阶段门均已通过；P5 已完成并交付，P6 未开始。
- 修复 commit：`2eaa273`（已随 P5 推送）。

### BUG-P5-001：Strict deterministic CUDA 不支持 DenseNet avg_pool3d backward

- 状态：`RESOLVED`
- 严重度：`HIGH`
- 发现日期：2026-08-11
- 影响阶段：P5
- 影响验收标准：曾影响；P5-R1 要求小数据 overfit 能降低 MSE，Stage A 必须完成 overfit 和 batch-16 preflight。旧 strict profile 在首次 optimizer step 前失败；warn-only remediation 已通过完整 Stage A。
- 恢复阶段：P5
- 受影响下游阶段：曾为 P6、P7、P8；修复已验证，三者继续保持 `NOT_STARTED`，不再标记为风险。
- 现象：H200 job `8964634.kman.restech.unsw.edu.au` 等待 `eligible_time=00:04:16` 后于 09:15 在 `k205` 的 NVIDIA H200 上启动；remote integrity verify 为 `PASS`。Train-only overfit backward 随后抛出 `avg_pool3d_backward_cuda does not have a deterministic implementation`，job `Exit_status=1`，preflight 未执行。
- 复现方式：使用 active H200 execution profile 在 CUDA 上运行 P5 Stage A overfit-check，并保持 `torch.use_deterministic_algorithms(True)`；DenseNet 的 3D average-pooling backward 触发该 RuntimeError。
- 根因：当前 PyTorch/CUDA build 对 `avg_pool3d_backward_cuda` 没有 strict deterministic implementation；全局 deterministic-algorithms enforcement 将该缺失实现作为错误阻断 backward。
- 修复：commit `11658ab` 创建独立 H200 warn-only execution/reproducibility profile，保留 `torch.use_deterministic_algorithms(True)` 但设置 `warn_only=True`；P5 runtime、PBS、transfer manifest、audit provenance 和直接测试均从该 profile 读取并记录此 policy。该修复不修改 Baseline-v1/v2 frozen requirements/config，不改变 H200、数据、模型、loss、precision、optimizer、batch、augmentation、scheduler、checkpoint 或 evaluation protocol。完整测试 `170 passed`、P5 direct tests `35 passed`；KDM sync 和 remote integrity verify 已通过。
- 验证命令与结果：Katana job `8964634` 确认运行于 `k205`/NVIDIA H200，remote integrity 为 `PASS`，最终 `Exit_status=1`；错误发生在 overfit backward。重跑 job `8965003.kman.restech.unsw.edu.au` 随后在 `k220` 的 NVIDIA H200 GPU 2 以 `Exit_status=0` 完成：8 samples/40 steps overfit MSE 从 `0.1297724843` 降至 `0.0642339364`；true batch 16 forward、MSE、backward 与 Adam step 均为 true；peak reserved 为 `2,860,515,328 / 150,393,585,664 bytes = 1.9020%`，低于 85% 门槛。运行仅发出预期的 AvgPool3d 与 MaxPool3d 非确定性 warn-only warnings。该失败及修复均不是 GPU availability、queue allocation、CUDA OOM 或显存容量不足问题。
- 未解决事项：无。Stage A、Formal Fold 0、folds 1–4 与五折 OOF 阶段门均已通过；P5 已完成并交付，P6 未开始。
- 修复 commit：`11658ab`（已随 P5 推送）。

### BUG-V2M-001：Git checkout 不保留 tracked config 的 read-only mode bits

- 状态：`RESOLVED`
- 严重度：`LOW`
- 发现日期：2026-08-10
- 影响阶段：V2M
- 影响验收标准：否；config canonical bytes、SHA-256 与生成时 read-only 行为始终有效，仅 fresh checkout 的不可移植测试断言失败。
- 恢复阶段：V2M（已保持 `COMPLETED`）
- 受影响下游阶段：无
- 现象：Git checkout 可将 committed `baseline_v2.resolved.yaml` 与 `baseline_v2.sha256` 恢复为 `0644`，导致直接断言 tracked snapshot 必须为 `0444` 的测试在 checkout 后失败。
- 复现方式：在新的 Git checkout 中检查 tracked snapshot mode 并运行 `test_v2_resolved_config_and_digest_match_source`。
- 根因：Git index 对普通文件只持久化 executable bit，不持久化 owner/group/other 的 read-only mode bits。
- 修复：只移除 `tests/test_v2_protocol.py` 中对两个 tracked snapshot 的 `0444` mode 断言；不修改 `freeze_config`、V2 config、resolved bytes、digest 或科学协议。
- 验证命令与结果：V2 专项测试 `26 passed`；完整测试 `86 passed`；独立 Phase Compliance Reviewer 复核为 `PASS`。`test_config` 继续验证 `freeze_config` 新生成的 resolved/hash 文件为 `0444`，V2 protocol test 继续验证 canonical bytes 与 committed SHA-256。
- 未解决事项：无。
- 修复 commit：`f28484f`。

### BUG-P3-001：single-slice source crop 的 QA contour rendering 失败

- 状态：`RESOLVED`
- 严重度：`MEDIUM`
- 发现日期：2026-08-10
- 影响阶段：P3
- 影响验收标准：是；影响 P3-R3 pilot QA 的完整 41-sample 可视化审阅，但不影响已写入 ROI 的 image/mask 内容。
- 恢复阶段：P3
- 受影响下游阶段：P4（该 Bug 处理期间保持 `NOT_STARTED`，未受实现影响）
- 现象：首次 pilot 的 41 个确定性样本中，33 个 QA 图成功生成；8 个 source crop 在 depth、height 或 width 方向只有一个 voxel，Matplotlib contour 对该 `1×N` 或 `N×1` plane 抛出 `TypeError`。
- 复现方式：对含单 voxel 轴的 non-empty source mask 调用 `_qa_image`；修复前会在 contour overlay 处失败。
- 根因：QA writer 未在调用 Matplotlib contour 前检查 plane 的两个空间维度是否均至少为 2。
- 修复：仅当 plane 两维均至少为 2 且 binary mask 非空时尝试绘制 contour；如果 Matplotlib 对退化 contour topology 仍抛出 `TypeError`，改用半透明 binary-mask overlay，确保 QA 不会因绘图失败而丢失可见 mask 证据。QA image 和 ROI 构建的其余逻辑不变。新增 single-slice 与 contour-raises visible-fallback 回归测试。
- 验证命令与结果：QA fallback 与 failure registry 修复后，`pytest -q tests/test_p3_roi.py` 为 `25 passed`，完整套件为 `111 passed`；冻结 V1/V2 requirements/config 无 diff。最终以修复后的 writer 完成 deterministic pilot 的 41/41 QA 图和 `verify --scope pilot` 的 41/41 ROI 验证，私有 failure registry 为 0 条。
- 未解决事项：无。用户 pilot alignment 确认仍为 full 前的阶段内人工门，不是该渲染 Bug 的未解决事项。
- 修复 commit：`8693bb9`、`86c0b8f`、`f72a01f`（均为本地 commit，尚未推送）。

### BUG-P3-002：private Parquet 使用固定临时文件名时存在重入写入冲突

- 状态：`RESOLVED`
- 严重度：`MEDIUM`
- 发现日期：2026-08-11
- 影响阶段：P3
- 影响验收标准：是；若多个 resumable full-build invocation 在同一 private index/failure-registry 路径使用固定临时名，可能使阶段进度持久化失败或覆盖，影响 P3 full ROI 的可恢复性证据。
- 恢复阶段：P3
- 受影响下游阶段：P4（该 Bug 处理期间保持 `NOT_STARTED`，未受实现影响）
- 现象：此前 `_write_index` 与 `update_private_failures` 均使用由目标文件名固定派生的 `.tmp` 路径；重入、并行或残留 temporary 情况下无法保证两个 writer 使用不同临时文件。
- 复现方式：保留 legacy fixed-name temporary file 后调用新的 private Parquet writer；回归测试确认 writer 不会读取、覆盖或删除该旧路径，并生成唯一 sibling temporary 后原子 replace。
- 根因：临时 Parquet 文件名不是 invocation-unique，不能满足 resumable full build 的安全持久化要求。
- 修复：新增共享 `_atomic_parquet`，以同目录 `tempfile.mkstemp` 创建唯一 temporary，并由 ROI index 与 failure registry 使用；`finally` 只清理当前 invocation 自己的 temporary。该修复不修改 ROI、mask、原始 DICOM、科学协议或 tracked audit 内容。
- 验证命令与结果：保留 `test_atomic_parquet_uses_unique_temporary_not_legacy_fixed_name`，并新增 mock 与实际第二进程的 `test_p3_build_lock_rejects_*` 回归测试。`a790e54` 使用 non-blocking exclusive `flock` 覆盖 private progress 的完整 read–merge–replace 生命周期；第二 writer 确定性得到 `P3_BUILD_ALREADY_RUNNING`，不会发生 last-writer-wins。最终 `verify --scope full` 为 2,633/2,633，P3 测试 `32 passed`、完整 `118 passed`，private failure registry 为 0，aggregate audit 为 2,633 个成功/非空 ROI。
- 未解决事项：无。
- 修复 commit：`72c4979`、`a790e54`（均为本地，尚未推送）。

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

### DIF-V2M-001：冻结 CUDA 环境与 Blackwell GPU architecture 不兼容

- 状态：`RESOLVED`
- 所属阶段：V2M
- 首次记录：2026-08-10
- 解决日期：2026-08-10
- 原影响：Katana 首次 V2 CUDA smoke job `8960330.kman.restech.unsw.edu.au` 分配到 NVIDIA RTX PRO 6000 Blackwell Server Edition；冻结的 PyTorch 2.5.1+cu121 build 不含 `sm_120` kernels，作业 Exit 1，无法形成 CUDA smoke 通过证据。
- 当前结论：不改变冻结依赖环境，改为在 V2M PBS request 中固定兼容的 `gpu_model=L40S`。最终 job `8960395.kman.restech.unsw.edu.au` 在 NVIDIA L40S 上 Exit 0，CUDA smoke 为 `PASS`。
- 解决证据：`artifacts/baseline_v2/audit/v2m/katana_cuda_jobs.json` 保存两次尝试和最终 policy；`cuda.json` 记录 L40S、PyTorch 2.5.1+cu121、MSE、unconstrained linear output、同一 V2 config hash与有限非零 gradients。
- 解除条件：已满足；兼容 GPU request 已固定并由自动测试覆盖。
- 关联 Bug：无；这是冻结软件与调度所得新 GPU architecture 的兼容性困难。

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
- 当前结论：不阻塞 P0–P5。P5 已完成五折 H200 formal runs 与 CPU OOF audit；当前容量足以保存 Black-box private runs，但该证据仍不能覆盖 P6–P10 的全部模型、Grad-CAM 和解释产物。
- 缓解措施：不上传原始 DICOM；通过 KDM 传输；正式 job 使用 `$TMPDIR`；重要数据和证据保留本地副本。
- P3 测量证据：P3 technical gate 已生成 2,633 个私有 ROI，合计 `1,002,688,586` bytes（约 `0.93 GiB`）；不含 private manifest、future checkpoints、predictions、Grad-CAM 和 `$TMPDIR` 运行时空间。
- P4 远程证据：explicit KDM workset 为约 `1.2 GiB`，Katana scratch 为 128 GiB total / 7.6 GiB used / 121 GiB available；job `8962963.kman.restech.unsw.edu.au` 已在 L40S 上 Exit 0。该证据满足 P4 smoke，但不代表 P10 正式实验工作集已完成估算。
- P5 测量证据：五折 Black-box private run files 合计 `1,360,388,058` bytes / 57 files；每折包含 `best.pt`、`last.pt`、80-epoch history、test predictions、metrics、runtime、test transaction evidence 和 plots。Private OOF Parquet 仅保留在 Katana，SHA-256 为 `6f7e8b840638cfcce3427a1a1e63155860f1067ac6d09f10e7c43aa74a2763e8`。该证据可用于后续估算，但仍不能代表 P6–P8 三种 concept models、P9 Grad-CAM/intervention 或全部解释产物的总工作集。
- 下一步：等待扩容回复；在 P10 前以已测量 ROI 大小加上 checkpoint、predictions、contributions、Grad-CAM 和临时文件估算正式总工作集。
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
- 交付状态：`11e164e` 已 fast-forward 合并至 `main` 并推送 GitHub；本地 `main` 与 `origin/main` SHA 已核对一致，合并后的完整测试为 `64 passed`。

### V2M 完成记录

- 完成日期：2026-08-10
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：创建并冻结 Baseline-v2 requirements、protocol index、source/resolved config 与 SHA-256；将 active protocol 治理切换至 Baseline-v2；实现 unconstrained linear regression task interface、V2 cohort rematerialization、CPU/MPS/CUDA smoke 和 Katana CUDA job；保存 versioned 脱敏审计证据。
- 科学协议结论：四模型只使用一个未裁剪连续 regression task head，`output_activation=none`、`output_constraint=unbounded`，直接以 normalized malignancy target 训练 MSE；不存在独立 binary head，secondary extreme binary evaluation 复用同一连续 score。
- Cohort 证据：2,634 个 stable physical clusters 的 `nodule_uid` 集合与顺序均保持不变；2,633 primary regression nodules / 868 patients；1,073 secondary extreme nodules / 578 patients，其中 782 low、291 high；1,560 middle-spectrum nodules 保留在 primary；1 个 missing-required-target cluster 排除。
- 配置与环境证据：V2 config SHA-256 为 `07ad34dc3449383bb195d126d6fedc1db3428198b2144fa75dc38fad939c33ce`；CPU、MPS 和 Katana CUDA audit 均为 `PASS`。首次 Blackwell job `8960330` 的 `sm_120` incompatibility 已通过固定兼容 L40S request 解决，最终 job `8960395.kman.restech.unsw.edu.au` Exit 0。
- Config portability 证据：Git 只持久化 executable bit，因此 tracked resolved/hash snapshots 在 checkout 后不以 `0444` 为可移植契约；`freeze_config` 生成时的 `0444` 行为仍由 `test_config` 验证，冻结完整性继续由 canonical resolved bytes 与 committed SHA-256 保护。
- 验收标准与证据：V2M-R1–V2M-R5 `PASS`；完整测试 `86 passed`；Phase Compliance Reviewer `PASS`；Status Synchronization Reviewer `UPDATED`；用户于 2026-08-10 明确确认。
- 产物路径：`docs/LIDC_IDRI_BASELINE_V2_REQUIREMENTS.md`、`docs/PROTOCOL_INDEX.md`、`configs/baseline_v2.*`、`artifacts/baseline_v2/audit/`、本地 ignored 的 `artifacts/baseline_v2/manifests/nodules.parquet`、`src/lidc_baseline/regression.py`、`src/lidc_baseline/v2_migration.py`、`scripts/katana/v2_cuda_smoke.pbs` 和对应测试。
- Baseline-v1 保护：V1 requirements/config/resolved/hash 无任何 diff；Baseline-v1 继续为 `SUPERSEDED/audit-only`。
- 已解决 Bug：`BUG-V2M-001`。
- 已解决困难：`DIF-V2M-001`。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`，不影响 V2M 完成结论。
- 明确未纳入内容：P3 consensus mask/ROI、split、模型训练或 P4–P10 实现；P3 保持 `NOT_STARTED`，实施计划尚未制定。
- 阶段门结论：`PASS`
- V2M commits：`b4155c3`、`5d8dffd`、`f07067e`、`51528ea`、`8d9dc7b`、`8007319`；完成状态 commit：本记录所在 commit。
- 交付前修复 commit：`f28484f`（`BUG-V2M-001`）。
- 交付验收：完成记录、portability fix 与状态记录均位于 `main`；本地 `main` 与 `origin/main` SHA 一致后才关闭本次交付。

### P3 完成记录

- 完成日期：2026-08-11
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：50% consensus mask、projection-sorted CT volume alignment、P1 exact-duplicate selection、tight bbox→cube high-side padding→fixed `64³` trilinear/nearest ROI、deterministic private NPZ、private ROI index/failure registry、41-sample pilot QA、private full ROI build、脱敏 aggregate audit，以及可恢复 single-writer full-build persistence。
- Cohort 与 QA 证据：2,633/2,633 primary ROI 均已写入并通过 full verify；2,633 non-empty binary masks、0 private failures、876 CT series、1 次 exact-duplicate policy 应用、0 个原始 DICOM 修改。41 个 deterministic pilot QA 图覆盖 reader counts 1–4、最小/最大 physical-volume candidates 与 exact-duplicate sample，且已获用户人工对齐确认。
- 验收标准与证据：P3-R1–P3-R3 与冻结协议保护均为 `PASS`；`p3_roi verify --scope full` 为 2,633/2,633；P3 专项 `32 passed`、完整 `118 passed`；脱敏 audit 记录 ROI 总量 `1,002,688,586` bytes（约 0.93 GiB）、bbox/cube/padding/resize-volume 分布、reader-count distribution 与 reconciliation 2,633→2,633；Phase Compliance Reviewer `PASS`、Status Synchronization Reviewer `UPDATED`；用户已于 2026-08-11 明确确认。
- 产物路径：`src/lidc_baseline/p3_roi.py`、`tests/test_p3_roi.py`、`artifacts/baseline_v2/audit/p3/`；ROI、QA 图、private manifest/index/failure registry 保持本地 Git ignored。
- 已解决 Bug：`BUG-P3-001`（single-slice QA contour rendering）与 `BUG-P3-002`（private Parquet temporary/reentrant persistence）；后者使用 unique sibling temporary、exclusive `flock` 和实际第二进程回归测试保证 single-writer lifecycle。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`；已测量 ROI 为约 0.93 GiB，但 P10 前仍须估算 checkpoints、predictions、contributions、Grad-CAM 与临时文件，且可用 Katana storage 必须达到预计工作集的 120%。
- 明确未纳入内容：P4 patient-level split、shared initialization、任何模型/训练或 P4–P10 开发；截至 P3 完成记录形成时，P4 保持 `NOT_STARTED`。
- 阶段门结论：`PASS`
- P3 commits：`0575bcf` 至 `dc8c356` 的 P3 implementation/status commits。
- 交付状态：已 fast-forward 合并至 `main` 并推送 GitHub；本地 `main`、`origin/main` 与 `HEAD` 已核对为同一 `dc8c356`，合并后完整测试为 `118 passed`。

### P4 完成记录

- 完成日期：2026-08-11
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 已完成内容：为 Baseline-v2 primary regression cohort 建立 deterministic patient-grouped five-fold outer/inner splits、严格 train-only statistics boundary、每折唯一且由四个模型共享的 DenseNet-121 encoder initialization，以及 P4 Katana KDM transfer 和 L40S CUDA integrity/forward smoke；未执行 optimizer、backward、parameter update 或模型训练。
- Cohort 与 split 证据：2,633 nodules / 868 patients，patient leakage 为 0，pooled outer-test 精确覆盖 2,633 nodules / 868 patients。Fold 0 train/validation/test 为 `1882/611, 272/86, 479/171` nodules/patients；fold 1 为 `1858/602, 273/86, 502/180`；fold 2 为 `1853/612, 241/87, 539/169`；fold 3 为 `1813/608, 271/86, 549/174`；fold 4 为 `1811/607, 258/87, 564/174`。所有 validation/test folds 均含 low/high extremes，四模型共用相同 split artifacts。
- Train-only statistics 证据：每折 normalized malignancy、六个 continuous concepts、两个 categorical vote distributions、valid-reader counts 和 train nodule-set fingerprint 只由 train membership 计算；validation、test、unknown UID、source hash 和 patient leakage guards 均通过。
- Shared initialization 证据：五折各生成一个 deterministic legacy-serialized private encoder artifact；每折 Black-box、Standard CBM、CEM 和 GAM 四个独立 consumers 加载后的 semantic encoder hash 完全一致，不同 folds 的 initialization hashes 不同，artifact provenance/corruption/overwrite guards 均通过。
- Katana 证据：通过 KDM explicit whitelist 同步 2,667 个路径，其中 2,666 个 hashed entries 共 `1,233,219,041` bytes；不含原始 DICOM/XML、Git metadata、QA 图、reports 或 runs。PBS job `8962963.kman.restech.unsw.edu.au` 在 NVIDIA L40S 上 `Exit_status=0`、walltime `00:01:47`；2,633 ROI integrity、五折 split/source hashes、每折四 consumer hashes及真实 ROI CUDA forwards 均为 `PASS`，且 `optimizer_created=false`、`backward_called=false`、`parameter_update=false`。
- 验收标准与证据：P4-R1–P4-R3 为 `PASS`；P4 prepare+Katana `17 passed`、合并前后完整测试均为 `135 passed`、本地与合并后 `p4_prepare verify` 和 frozen V1/V2 requirements/config/resolved/hash 检查均通过；阶段级 Phase Compliance Reviewer、Status Synchronization Reviewer、completion-sealing 与 post-delivery Phase Compliance Reviewers 均通过；用户已于 2026-08-11 明确确认 P4。
- 产物路径：`src/lidc_baseline/p4_prepare.py`、`src/lidc_baseline/p4_katana.py`、`scripts/katana/sync_p4.sh`、`scripts/katana/p4_cuda_smoke.pbs`、`tests/test_p4_prepare.py`、`tests/test_p4_katana.py`、`artifacts/baseline_v2/audit/p4/`；private manifest、ROI index、splits、encoder initializations 与 transfer manifest 保持 Git ignored。
- 证据 anchors：tracked audit evidence 为 `9d24035`；approval-gate verified anchor 为 `e0634e7`；delivery anchor 为 `ec7bd8e`。
- 已解决 Bug：无。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`，不阻止 P4 完成；P10 前仍须估算正式训练、checkpoint、prediction、contribution、Grad-CAM 和临时文件总工作集。
- 明确未纳入内容：Black-box model、optimizer、训练、checkpoint selection 或任何 P5–P10 实现；P5 保持 `NOT_STARTED`。
- 阶段门结论：`PASS`
- 交付状态：P4 已 fast-forward 合并至 `main` 并推送 GitHub。交付时 `main`、`HEAD` 与 `origin/main` 均为 `ec7bd8e89528ae4adeea4699217fc481f402e0c7`；合并后完整测试为 `135 passed`，`p4_prepare verify` 与 post-delivery Phase Compliance Reviewer 均为 `PASS`。P5 保持 `NOT_STARTED`，须另行制定并批准实施计划。

### P5 完成记录

- 完成日期：2026-08-11
- 生命周期：`COMPLETED`
- 健康状态：`ON_TRACK`
- 用户确认：用户于 2026-08-11 明确批准 P5 五折 Black-box regression 科学执行、OOF reconciliation、审计和阶段门结果。
- 已完成内容：在 P4 固定的 patient-grouped splits 与 shared DenseNet-121 encoder initializations 上，以 frozen H200 warn-only execution profile 完成 Black-box unconstrained linear malignancy regression。训练固定 Adam、`lr=1e-4`、true batch 16、80 epochs、FP32/no-AMP/no-BF16/no-TF32、train-only deterministic augmentation 与 validation-MSE checkpoint rule；没有 concept inputs、mask input 或 independent binary head。
- Stage A 与五折执行证据：H200 Stage A job `8965003` 完成 overfit 和 true-batch-16 forward/MSE/backward/Adam preflight。Formal Fold 0 job `8965243` 完成科学训练、best checkpoint 和 test exactly once；其 PBS final verifier false failure 由 `BUG-P5-002` 修复后仅对 existing artifacts 重新验证为 `PASS`，未重训或重复 test。Stage B jobs Fold 1=`8965994`、Fold 2=`8965995`、Fold 3=`8965996`、Fold 4=`8965997` 均完成 80 epochs、minimum-validation-MSE checkpoint、test exactly once 与 final verifier，且 `Exit_status=0`。
- Checkpoint 证据：五折 best epoch indexes 为 `[14,19,10,15,38]`，对应 validation MSE 为 `[0.0199759813899636,0.0247182929771812,0.0238741965997411,0.0228595164578849,0.0193156898568515]`；每折均使用 P4 指定 encoder initialization 与 deterministic fold-specific linear-head seed/hash，test 不参与 checkpoint selection。
- OOF 与 pooled metrics：CPU-only job `8966614` 在 `k189` 以 `Exit_status=0` 完成。OOF 精确覆盖 2,633 nodules / 868 patients，fold counts `[479,502,539,549,564]`，每个 nodule 恰有一次 test prediction，patient leakage 为 0。Pooled original-scale MAE/RMSE=`0.50062578502153/0.6421887532919749`，normalized MAE/RMSE=`0.1251564462553825/0.16054718832299372`，Pearson/Spearman=`0.7157050987783329/0.634502488281551`，prediction min/max=`-0.12787137925624847/1.0300079584121704`，below-0/above-1 rates=`0.014812001519179644/0.001139384732244588`。
- Private 与 tracked 产物：private `oof_predictions.parquet` 仅保留在 Katana，SHA-256 为 `6f7e8b840638cfcce3427a1a1e63155860f1067ac6d09f10e7c43aa74a2763e8`；五折 private run storage 为 `1,360,388,058` bytes。仓库仅保存 `artifacts/baseline_v2/audit/p5/fold_0.json`–`fold_4.json` 与 `summary.json` 六个脱敏 audit JSON，不保存 private predictions、checkpoints 或 patient/nodule identifiers。
- 验收标准与证据：P5 audit/Katana direct tests `8 passed`、完整 tests `173 passed`；五折 final verifier、OOF set equality、patient isolation、execution/config/encoder/head hashes、unconstrained output 与 one-time-test invariants 均为 `PASS`。Phase Compliance Reviewer 为 `PASS`，Status Synchronization Reviewer 为 `UPDATED`，冻结 V1/V2 requirements/config 无 diff。
- 已解决 Bug：`BUG-P5-001`（PyTorch CUDA 3D pooling backward 缺少 strict deterministic implementation；独立 warn-only execution profile 修复）与 `BUG-P5-002`（JSON/CSV validation-MSE round-trip exact-equality false failure；`math.isclose` verifier-only 修复）。两项均重新验收通过，没有改变数据、模型、loss、optimizer、batch、augmentation、checkpoint 或 evaluation protocol。
- 关键 commits：OOF implementation `a81d06b`、脱敏 audit evidence `0359d61`、approval-gate status `6e07bb5`、completion/delivery `147f8f0`、post-delivery 状态同步 `c392c04`；均已进入 `main` 并推送。
- 明确未纳入内容：P6 Standard CBM、P7 CEM、P8 GAM、P9 bootstrap/Youden-J/intervention/Grad-CAM 与 P10 最终报告均未实现；P6 保持 `NOT_STARTED`。
- 遗留困难：`DIF-P10-001` 继续为 `OPEN`，不影响 P5 完成；P5 已提供 `1,360,388,058` bytes 的五折 Black-box private run 测量证据，P10 前仍需估算其余模型和解释产物总工作集。
- 阶段门结论：`PASS`
- 完成状态 commit：`147f8f0`。
- 交付状态：P5 已 fast-forward 合并至 `main` 并推送 GitHub；合并后完整测试为 `173 passed`，冻结 V1/V2 requirements/config 无 diff，post-delivery Phase Compliance Reviewer 为 `PASS`。Post-delivery 状态同步 commit `c392c04` 也已推送；最终核验时本地 `main`、`HEAD` 与 `origin/main` 均为 `c392c04c556a563c4b1fefd6ae69c3735c742083`。

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
| 2026-08-09 | `PHASE_COMPLETED` | P2 | 用户确认 P2；永久记录已保存 cohort、provenance、审计与验收证据；已 fast-forward 合并至 main 并推送，P3 保持未开始 | `11e164e` |
| 2026-08-10 | `PROTOCOL_MIGRATION_STARTED` | V2M | 用户批准 Baseline-v2 连续评分协议与 unconstrained linear regression output；开始 V2 requirements/config、smoke 和 cohort rematerialization，P3 保持未开始 | `v2-protocol-migration` 本地分支 |
| 2026-08-10 | `PHASE_AWAITING_APPROVAL` | V2M | V2 requirements/config、CPU/MPS/CUDA smoke、cohort rematerialization、86 项测试与阶段级 Phase Compliance Reviewer 均为 `PASS`；5 个 V2M commits 已保存于本地但未推送，等待用户确认，P3 保持 `NOT_STARTED` | `b4155c3`、`5d8dffd`、`f07067e`、`51528ea`、`8d9dc7b`（local, unpushed） |
| 2026-08-10 | `PHASE_COMPLETED` | V2M | 用户确认 V2M；Baseline-v2 active protocol、配置、linear-regression smoke、cohort rematerialization 和验收证据已永久封存，GitHub 交付按确认后流程执行；P3 保持 `NOT_STARTED` | 本次 V2M 完成状态提交 |
| 2026-08-10 | `BUG_RESOLVED` | V2M | 交付前发现 Git checkout 不持久化 tracked `0444` mode；仅移除不可移植测试断言，保留生成时 `0444`、canonical bytes 和 SHA-256 保护。V2M 保持 `COMPLETED`，P3 保持 `NOT_STARTED` | `f28484f` |
| 2026-08-10 | `PHASE_STARTED` | P3 | 用户批准 P3 consensus mask、确定性 `64³` ROI 与 pilot-first QA 实施计划；P3 进入 `IN_PROGRESS`，P4 当时保持 `NOT_STARTED` | `p3-consensus-roi` 本地分支 |
| 2026-08-10 | `BUG_RESOLVED` / `PILOT_QA_COMPLETE` | P3 | `BUG-P3-001` 修复后，deterministic pilot 41/41 ROI、QA 图和 verify 均通过，failure registry 为 0；P3 当时仍为 `IN_PROGRESS`，必须等待用户确认 pilot alignment，P4 当时保持 `NOT_STARTED` | 本次状态同步提交 |
| 2026-08-10 | `PILOT_QA_APPROVED` | P3 | 用户明确确认 41 个 deterministic pilot QA 图配准正确；仅授权构建 P3 full 私有 ROI/index 和脱敏 aggregate audit，不授权进入 P4、P3 阶段完成或推送 | 用户确认 |
| 2026-08-11 | `FULL_BUILD_OUTPUTS_WRITTEN` / `BUG_VERIFYING` | P3 | private full ROI/index 的当时 outputs 与 local aggregate audit 均报告 2,633/2,633、0 private failures；当时尚未完成 full verify、完整测试或阶段门。`72c4979` 已创建唯一 temporary 的 atomic Parquet 修复与回归测试，仍待最终验证；P3 当时保持 `IN_PROGRESS`，P4 当时保持 `NOT_STARTED` | `72c4979`（本地，未推送） |
| 2026-08-11 | `PHASE_AWAITING_APPROVAL` / `BUG_RESOLVED` | P3 | full verify 为 2,633/2,633，aggregate audit 为 2,633 个成功/非空 ROI、0 failures、876 个 CT series、1 次 exact-duplicate policy；P3 tests `32 passed`、完整 tests `118 passed`。`a790e54` 的 exclusive `flock` 已以实际第二进程回归测试验证，`BUG-P3-002` 关闭。P3 技术阶段门和双 agent 审查均通过，当时正在等待用户最终确认；P4 当时保持 `NOT_STARTED`。 | `a790e54`、`ac5c9ec`（本地，未推送）；本次状态同步提交 |
| 2026-08-11 | `PHASE_COMPLETED` / `DELIVERED` | P3 | 用户明确确认 P3；P3 永久记录已写入 full ROI/QA/audit、测试、Bug 与 storage 证据，并已 fast-forward 合并、推送至 GitHub。`main`、`origin/main` 与 `HEAD` 当时均为 `dc8c356`；P4 在该交付记录形成时保持 `NOT_STARTED`。 | `dc8c356` |
| 2026-08-11 | `PHASE_STARTED` | P4 | 用户批准 patient-grouped five-fold split、train-only statistics、每折 shared DenseNet initialization 与 Katana L40S loading/hash smoke 计划；P4 进入 `IN_PROGRESS`，P5 保持 `NOT_STARTED`。 | `p4-splits-shared-init` 本地分支 |
| 2026-08-11 | `LOCAL_IMPLEMENTATION_VERIFIED` | P4 | 本地 P4 implementation/tests 已完成真实 private build/verify：2,633 nodules / 868 patients、五折固定 counts、exact ROI file hashes、train-only statistics 与每折四 consumer shared encoder hashes 均通过；P4 `9 passed`、完整 `127 passed`，当前批次 Phase Compliance Reviewer `PASS`。Katana 同步/L40S smoke 与 tracked aggregate audit 尚未完成，因此 P4 保持 `IN_PROGRESS`，P5 保持 `NOT_STARTED`。 | `6b2342f`（本地，未推送） |
| 2026-08-11 | `PHASE_AWAITING_APPROVAL` | P4 | 实际 KDM whitelist sync 与 L40S job `8962963.kman.restech.unsw.edu.au` 均通过；2,633 ROI / 868 patients、五折 split、每折四 consumer initialization hashes、真实 ROI CUDA forwards 和 no-training invariants 均验证。Tracked audit evidence、P4 `17 passed`、完整 `135 passed`、冻结协议检查与阶段级双 agent 审查均为 `PASS`。Evidence 与 approval-gate anchors 分别为 `9d24035`、`e0634e7`；全部 P4 工作保存在本地分支且未推送，当前仅等待用户确认，P5 保持 `NOT_STARTED`。 | `9d24035`、`e0634e7` |
| 2026-08-11 | `PHASE_COMPLETED` | P4 | 用户明确确认 P4；completion-sealing Phase Compliance Reviewer 为 `PASS`，P4 永久记录已保存 split、train-only statistics、shared initialization、KDM、Katana、测试和双审查证据。P4 本地分支尚待 fast-forward 合并、`main` 测试和推送；P5 保持 `NOT_STARTED`。 | 用户确认与 P4 完成记录 |
| 2026-08-11 | `DELIVERED` | P4 | P4 已 fast-forward 合并至 `main` 并推送 GitHub；合并后完整测试 `135 passed`、`p4_prepare verify` 与 post-delivery Phase Compliance Reviewer 均为 `PASS`。交付时 `main`、`HEAD` 与 `origin/main` 三方 SHA 一致；P5 保持 `NOT_STARTED`。 | `ec7bd8e` |
| 2026-08-11 | `PHASE_STARTED` | P5 | 用户批准 Reference-aligned Black-box Regression 两阶段计划及 Fold-0 前四项实现澄清；P5 进入 `IN_PROGRESS`，先执行 common config、实现/测试、overfit、L40S preflight 与 Fold 0 formal gate。Fold 0 中间确认前禁止 folds 1–4；P6 保持 `NOT_STARTED`。 | `p5-blackbox-regression` 本地分支；基线 `960e366` |
| 2026-08-11 | `BUG_DISCOVERED` / `PHASE_BLOCKED` | P5 | H200 job `8964634` 在 `k205`/NVIDIA H200 上启动且 remote integrity `PASS`，但 overfit backward 因 strict deterministic algorithms 下 `avg_pool3d_backward_cuda` 无 deterministic implementation 而 Exit 1。登记 `BUG-P5-001`，切换 `BUG_MAINTENANCE / FULL_DOCUMENT`；P5 为 `BLOCKED / AT_RISK`，P6–P8 保持 `NOT_STARTED / AT_RISK`，未启动 formal Fold 0。 | Katana job `8964634` |
| 2026-08-11 | `BUG_FIXING` | P5 | 本地未推送 commit `11658ab` 为 `BUG-P5-001` 封存独立 H200 warn-only execution/reproducibility profile：继续启用 deterministic-algorithms enforcement，但对缺少 deterministic CUDA implementation 的操作记录 warning 而非阻断 backward；FP32、TF32-off、AMP/BF16-off、batch 16、H200 与其余 P5 protocol 不变。resolved SHA-256 为 `66c925a7b43bf9fa312ceb850b43746a34d1808888667c39392eaef9e47495bb`，本地 P5 delta manifest 为 7 files / `94,596` bytes、SHA-256 `fa1acbe51a3e15a0c78212c4ce7c6365dbde318959bfc249e5f76b4a0dc3c36b`。完整测试 `170 passed`、P5 direct tests `35 passed`、修复 diff 的 Phase Compliance Reviewer 为 `PASS`；尚未 KDM sync、未重跑 Stage A，故 P5 仍 `BLOCKED`、P6 未开始。 | `11658ab`（本地未推送）；尚无新的 Katana job |
| 2026-08-11 | `BUG_RETRY_QUEUED` | P5 | warn-only profile 的 KDM sync 与新的 remote integrity verify 均为 `PASS`；重跑 H200 Stage A job `8965003.kman.restech.unsw.edu.au` 已提交，显式请求 `gpu_model=H200`。当前 job 为 `Q`，scheduler comment 为 `Insufficient amount of resource: ngpus`，estimated start 为 2026-08-11 10:48:47；尚未执行 overfit、preflight 或 Fold 0，P5 继续 `BLOCKED`、P6 保持 `NOT_STARTED`。 | Katana job `8965003` |
| 2026-08-11 | `BUG_STAGE_A_PASS` / `BUG_VERIFYING` | P5 | H200 Stage A job `8965003.kman.restech.unsw.edu.au` 在 `k220` NVIDIA H200 GPU 2 以 `Exit_status=0` 完成；8 samples/40 steps overfit MSE 从 `0.1297724843` 降至 `0.0642339364`，true batch 16 forward/MSE/backward/Adam step 均通过，peak reserved 为 `2,860,515,328 / 150,393,585,664 bytes = 1.9020%`（低于 85%）。运行记录预期的 AvgPool3d 与 MaxPool3d warn-only warnings。P5 不再因 Stage A 技术阻断，但继续 `BUG_MAINTENANCE / VERIFYING`；完成合规复审并向用户报告确认前禁止启动 formal Fold 0，P6 保持 `NOT_STARTED`。冻结 V1/V2 requirements/config 未修改。 | Katana job `8965003` |
| 2026-08-11 | `BUG_RESOLVED` / `P5_RESUMED` | P5 | `BUG-P5-001` 的 warn-only remediation、H200 Stage A 回归与 Bug 修复 Phase Compliance Reviewer 均为 `PASS`；用户明确批准恢复 P5。状态恢复为 `NORMAL_DEVELOPMENT / CURRENT_AND_NEXT`，P5 为 `IN_PROGRESS / ON_TRACK`，仅授权启动 formal Fold 0。folds 1–4 与 P6 仍未获授权，冻结 V1/V2 requirements/config 未修改。 | `11658ab`（local, unpushed）；Katana job `8965003`；用户确认 |
| 2026-08-11 | `FORMAL_FOLD0_QUEUED` | P5 | 用户批准继续 P5 后，唯一获授权的 formal Fold 0 job `8965243.kman.restech.unsw.edu.au` 已提交。只读核验显示其为 `Q`，队列 `csegpu48`，显式请求一张 H200（`ngpus=1`、`gpu_model=H200`、`cpu_per_gpu_gte_8=1`、`mem=46gb`、`walltime=48:00:00`）。尚未开始训练、best checkpoint 或 test；folds 1–4 与 P6 继续禁止。 | Katana `qstat -fx 8965243` |
| 2026-08-11 | `FOLD0_RUN_COMPLETE` / `BUG_DISCOVERED` / `PHASE_BLOCKED` | P5 | Job `8965243` 在 `k205` H200 GPU 7 完成 80 epochs，best epoch index `14` / validation MSE `0.01997598138996362`，并在 checkpoint 固定后对 479 samples 完成唯一一次 test；scientific execution 为 `PASS`，original-scale MAE/RMSE=`0.4985995429/0.6436540272`。最终 verifier 因 JSON `0.01997598138996362` 与 CSV round-trip `0.0199759813899636` 的约 `2.08e-17` 表示差触发 `P5_BEST_OBJECTIVE_MISMATCH`，PBS Exit 1。登记 `BUG-P5-002` 并切换 `BUG_MAINTENANCE / FULL_DOCUMENT`；P5 为 `BLOCKED / AT_RISK`。按用户要求停止，不修复、不重训、不重复 test；folds 1–4 与 P6 未开始。 | Katana job `8965243`；Phase Compliance Reviewer `FAIL` |
| 2026-08-11 | `BUG_RESOLVED` / `FOLD0_GATE_PASS` / `STAGE_B_AWAITING_APPROVAL` / `P5_RESUMED` | P5 | 用户批准的 verifier-only commit `2eaa273` 使用 `rel_tol=1e-12, abs_tol=1e-12` 修复 JSON/CSV float round-trip false failure，并加入 tiny-difference positive 与 real-mismatch negative tests。P5 direct/full tests 为 `33 passed` / `172 passed`，Phase Compliance Reviewer 为 `PASS`；KDM transfer 与 remote integrity 为 `PASS`。只对既有 Fold 0 artifacts 重新运行 final verifier并返回 `PASS`，未重训、未重复 test，artifact hashes/mtimes 不变。`BUG-P5-002` 关闭，P5 恢复 `NORMAL_DEVELOPMENT / IN_PROGRESS / ON_TRACK`；当前等待用户一次性批准 Stage B folds 1–4，P6 保持 `NOT_STARTED`。 | `2eaa273`（本地未推送）；existing-artifact remote verify `PASS` |
| 2026-08-11 | `STAGE_B_APPROVED` / `FOLDS_1_4_QUEUED` | P5 | 用户明确批准一次性提交 Stage B folds 1–4。Fold 1=`8965994`、Fold 2=`8965995`、Fold 3=`8965996`、Fold 4=`8965997`，均以 `P5_STAGE_B_APPROVED=1` 提交至 `csegpu48`，显式请求 H200×1、8 CPU、46 GB RAM、48 小时 walltime，当前均为 `Q`。Phase Compliance Reviewer 为 `PASS`；只读检查确认用户远程 jobs 仅有这四个、无重复提交。Fold 0 未修改，P6 保持 `NOT_STARTED`；不得将排队状态表述为训练完成。 | Katana jobs `8965994`–`8965997`；用户批准 |
| 2026-08-11 | `PHASE_AWAITING_APPROVAL` / `FIVE_FOLD_OOF_PASS` | P5 | Stage B jobs `8965994`–`8965997` 均完成 80 epochs、minimum-validation-MSE checkpoint、test exactly once 与 final verify，且 `Exit_status=0`。CPU OOF job `8966614` 在 `k189` Exit 0；OOF 精确覆盖 2,633 nodules / 868 patients，fold counts `[479,502,539,549,564]`，patient leakage 为 0。六个 tracked P5 audit JSON 已通过脱敏检查；private OOF 仅保留在 Katana，SHA-256 `6f7e8b840638cfcce3427a1a1e63155860f1067ac6d09f10e7c43aa74a2763e8`。Direct `8 passed`、full `173 passed`、Phase Compliance Reviewer `PASS`、Status Synchronization Reviewer `UPDATED`；P5 转为 `AWAITING_USER_APPROVAL / ON_TRACK`，仅等待用户最终确认，P6 保持 `NOT_STARTED`，尚未合并或推送。 | `a81d06b`、`0359d61`（本地未推送）；Katana jobs `8965994`–`8965997`、`8966614` |
| 2026-08-11 | `PHASE_COMPLETED` | P5 | 用户明确确认 P5；五折 Black-box H200 formal runs、minimum-validation-MSE checkpoints、test exactly once、final verifies、2,633 nodules / 868 patients OOF、0 patient leakage、六个脱敏 audit、完整 `173 passed` 与双 agent 审查均已封存。P5 转为 `COMPLETED / ON_TRACK`，P6 保持 `NOT_STARTED`。Completion status commit、fast-forward merge、`main` 完整测试和 GitHub push 尚待执行，不得声称已交付。 | 用户确认；`a81d06b`、`0359d61`、`6e07bb5`；本次 completion status commit 待创建 |
| 2026-08-11 | `DELIVERED` | P5 | P5 已由 completion commit `147f8f0` fast-forward 合并至 `main` 并推送 GitHub；合并后完整测试 `173 passed`、冻结 V1/V2 requirements/config 无 diff，post-delivery Phase Compliance Reviewer 为 `PASS`。Post-delivery 状态同步 commit `c392c04` 也已推送；最终核验时本地 `main`、`HEAD` 与 `origin/main` 三方均为 `c392c04c556a563c4b1fefd6ae69c3735c742083`，P6 保持 `NOT_STARTED`。 | `147f8f0`、`c392c04` |
| 2026-08-11 | `PHASE_STARTED` | P6 | 用户批准 Sequential Standard CBM Regression 最终实施计划及四项澄清；P6 进入 `IN_PROGRESS / ON_TRACK`。Stage 2 只使用 frozen predicted concepts，task-head 输入固定为 activated 16D vector，test concept predictions 只在 task-best 固定后生成；Stage A 通过后可一次提交五个 H200 formal folds。P7 保持 `NOT_STARTED`。 | `p6-standard-cbm` 本地分支；基线 `9c8b60b` |
| 2026-08-11 | `LOCAL_CONFIG_VERIFIED` | P6 | P6 execution supplement、deterministic resolved config与SHA-256已生成并验证；resolved SHA-256为 `792f544aef33d30f122054ba40bdf8f185cea71e516614545ba3f85879ed3bc3`。配置专项测试 `4 passed`、完整测试 `177 passed`，Phase Compliance Reviewer `PASS`。该批次未实现concept model或训练lifecycle，未执行Stage A或提交正式作业；P6保持`IN_PROGRESS / ON_TRACK`，P7保持`NOT_STARTED`。 | `c3224f4`（local, unpushed） |
| 2026-08-11 | `LOCAL_CORE_PRIMITIVES_VERIFIED` | P6 | Standard CBM core primitives已实现八个独立linear heads、activated canonical 16D vector、八组等权loss、sample-weighted aggregation、deterministic head initialization、partition records/dataset、freeze/BatchNorm hashes、predicted-cache guard和两种量纲贡献重建。Direct tests `16 passed`、完整测试 `189 passed`，最终Phase Compliance Reviewer `PASS`。完整sequential lifecycle、cache persistence、test-once、Stage A与正式训练仍未完成；P6保持`IN_PROGRESS / ON_TRACK`，P7保持`NOT_STARTED`。 | `5aedec3`（local, unpushed） |
