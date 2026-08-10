# LIDC-IDRI Baseline-v2 需求文档

**状态：已明确批准并冻结**

**冻结日期：2026-08-10**

Baseline-v2 取代 Baseline-v1 成为后续开发和实验的唯一 active protocol。Baseline-v1 保持不可修改，仅供历史审计。Baseline-v2 的核心变化是将主要任务从 extreme-score 二分类改为完整 malignancy spectrum 的连续评分；同一连续输出同时用于 secondary extreme-score binary discrimination。

## 1. 项目目标与边界

在相同 physical-nodule cohort、patient-level splits、`64³` ROI、DenseNet-121 encoder architecture 和 fold-specific shared encoder initialization 下比较：

1. Black-box 3D DenseNet-121
2. Standard CBM
3. Mixed-type CEM
4. CBM + GAM

任务固定命名为：

> Radiologist-assessed pulmonary nodule malignancy scoring

Primary task：预测完整 LIDC reader-assessed malignancy spectrum。

Secondary task：在明确 low/high suspicion subset 上评估同一连续 score 的区分能力。

预测目标来自 LIDC-IDRI XML 中放射科医生的 malignancy ratings，不是病理确诊癌症标签。项目不是临床诊断系统。

核心比较维度：

- 连续 malignancy scoring 性能
- extreme-score binary discrimination
- concept prediction quality
- concept intervention effectiveness
- 8 组 concept contributions
- nodule-level spatial interpretability proxies

## 2. 全局实验协议

### G-R1：实验单位与数据隔离

一个样本对应一个 physical pulmonary nodule。Reader annotations 必须先通过确定性 clustering 聚合，再生成 target、concepts、consensus mask 和 ROI。

#### 验收标准

- 每个 stable `nodule_uid` 只对应一个 CT series 和一个 physical nodule。
- Reader annotations 可追溯至 canonical XML source。
- 同一 patient 的全部 CT、时间点和 nodules 始终属于同一 split。
- 不存在 annotation-level 或 patient-level leakage。

### G-R2：单头连续任务输出

每个模型只有一个 malignancy regression task head：

```text
target_normalized = (mean_malignancy - 1) / 4
raw_task_score = Linear(task_representation)
score_normalized = raw_task_score
score_1_to_5 = 1 + 4 * raw_task_score
```

Regression task head 最后一层必须是 unconstrained linear output：

- 不使用 sigmoid、tanh、clamp 或其他 output activation。
- `score_normalized` 是 normalized target scale 上的预测，不保证位于 `[0,1]`。
- `score_1_to_5` 不裁剪，允许暂时超出 `[1,5]`。
- 训练、checkpoint、metrics、intervention 和解释全部使用未裁剪输出。
- 输出不得称为 probability 或 malignancy logit。

该设计是 **Baseline-v2 pre-registered implementation choice**，不是 reference paper reported output head。

统一任务输出：

```text
malignancy_raw_score
malignancy_score_normalized
malignancy_score_1_to_5
```

前两个字段必须数值完全一致。

#### 验收标准

- 四模型均无独立 binary head。
- 模型图中不存在 regression task output sigmoid/tanh/clamp。
- 构造负值和大于 1 的输出时，接口不得静默裁剪。
- `1→0`、`3→0.5`、`5→1` target normalization 精确成立。
- Original-scale conversion 精确使用 `1 + 4s`。

### G-R3：模型输出范围

Black-box 只输出 malignancy score 与 malignancy Grad-CAM，不输出 concept artifacts。

Standard CBM、CEM、GAM 输出：

```text
malignancy scores
8-group concept predictions
8 raw normalized-scale contributions
8 centered normalized-scale contributions
normalized-scale centered bias
8 rating-point contributions
rating-scale centered bias
malignancy Grad-CAM
8 concept-specific Grad-CAMs
```

#### 验收标准

- Black-box 结果中不存在 concept placeholders。
- CBM/CEM/GAM 的 task prediction 只能读取 concept representation，不得直接读取 DenseNet feature。
- 每项结果可由 `nodule_uid`、fold、model、checkpoint 和 config hash 唯一追溯。

### G-R4：Primary regression 与 secondary extreme subset

对每个 physical nodule 的有效 reader malignancy ratings 取均值。

Primary regression：

- 只要 malignancy 与 8 个 concepts 各至少有一个有效 reader rating，即进入 primary regression cohort。
- `2 < mean_malignancy < 4` 不再从 primary task 排除。

Secondary extreme binary：

- `mean_malignancy <= 2`：label 0。
- `mean_malignancy >= 4`：label 1。
- `2 < mean_malignancy < 4`：secondary label 为 null，但仍参与 primary regression。

Patient Diagnoses XLS 不进入训练，不得覆盖 XML ratings。

#### 验收标准

- Primary cohort 为 2,633 nodules / 868 patients。
- Secondary subset 为 1,073 nodules / 578 patients，其中 782 low、291 high。
- 唯一 missing-required-target nodule 被排除并记录具体字段。
- Manifest 保存 raw ratings、valid-reader count、mean、normalized target、primary eligibility、extreme eligibility 和 extreme label。

### G-R5：Concept schema

固定 8 个 concept groups：

1. subtlety
2. internalStructure
3. calcification
4. sphericity
5. margin
6. lobulation
7. spiculation
8. texture

六个连续 concepts 使用合法范围归一化后的 reader mean；internalStructure 使用 4-class vote distribution；calcification 使用 6-class vote distribution。Malignancy 永远是 downstream target，不得作为输入 concept。

#### 验收标准

- 连续 concept targets 位于 `[0,1]`。
- Categorical distributions 的概率和在数值误差内为 1。
- 每个 target 的 raw ratings、valid-reader count 和 aggregated target 可审计。
- Concept input schema 不含 malignancy。

### G-R6：Categorical ties

True modal-label ties 保留用于 training 和 soft metrics，仅从 hard modal-label macro-F1 排除。

#### 验收标准

- Soft metric sample counts 包含 ties。
- Hard macro-F1 只使用 unique true modal label。
- 每 concept/fold/OOF 报告 tie 数和实际样本数。

### G-R7：协议冻结与产物版本

V2 使用独立 config hash 与产物命名空间。Baseline-v1 requirements/config/resolved/hash 不得修改。

#### 验收标准

- V2 requirements、source config、resolved config 和 SHA-256 一一对应。
- Split、checkpoint、prediction、metric 和 report 保存 V2 config hash。
- V1 与 V2 产物不得混写。
- 任何新科学变更创建新协议版本，不覆盖 V2。

## 3. 标准接口与产物

代码仓库：`/Users/katherine/Desktop/lidc_baseline`。

原始数据：`/Users/katherine/Desktop/lidc_data`。

V2 产物：

```text
configs/baseline_v2.yaml
configs/baseline_v2.resolved.yaml
configs/baseline_v2.sha256
artifacts/baseline_v2/manifests/nodules.parquet
artifacts/baseline_v2/audit/p2/
artifacts/baseline_v2/splits/fold_{0..4}.json
artifacts/baseline_v2/encoder_initializations/fold_{0..4}.pt
artifacts/baseline_v2/rois/{nodule_uid}.npz
runs/baseline_v2/{model}/fold_{k}/
reports/baseline_v2/
```

共享 P1 audit 继续位于 `artifacts/audit/p1/`，不复制原始 DICOM 或审计明细。

标准 ROI：

```text
image: float32 [1,64,64,64]
mask:  uint8   [1,64,64,64]
```

#### 验收标准

- 后续脚本只读取 V2 manifest/splits 和共享或 V2 ROI。
- V2 manifest 复用既有 stable `nodule_uid`，不得重新定义 physical nodules。
- 完整 manifest 保持本地私有；tracked audit 仅含脱敏摘要。
- 产物不依赖临时绝对路径。

## 4. 分阶段需求

### Phase 0：环境与 V2 配置复验

#### P0-R1：环境复用与 regression smoke

复用已冻结 Python、PyTorch、MONAI、pylidc 与 CUDA 环境。CPU、MPS、CUDA 使用合成 `[1,1,64,64,64]` image 和 normalized regression target，完成 DenseNet-121 linear output、MSE 和 backward。

##### 验收标准

- 三设备输出 shape 为 `[1,1]`，loss 和 gradients 有限且 gradients 非零。
- Audit 明确记录 `loss=mse`、`output_activation=none`、`output_constraint=unbounded`。
- Smoke 不读取原始 DICOM。
- 三设备记录同一个 V2 config hash。

#### P0-R2：确定性 V2 config

Config 必须记录 cohort、target normalization、linear output、concept schema、ROI、splits、shared encoder initialization、losses、checkpoint、metrics、contributions、interventions、bootstrap、Grad-CAM 和 occlusion。

##### 验收标准

- Canonical resolved YAML 重复生成字节一致。
- SHA-256 等于 resolved YAML bytes 的 digest。
- Regression task 配置不存在 sigmoid/tanh/clipping。

### Phase 1：共享 DICOM/XML 审计

#### P1-R1：Canonical source 与 mapping

复用已完成的 Baseline-v1 P1 audit：`LIDC-XML-only` 是 canonical source，DX/CR/CXR 不进入 cohort。

##### 验收标准

- 1,035/1,035 canonical CT XML 唯一映射，1,018/1,018 CT series covered。
- V2 不改变 P1 source mapping 或原始数据。

#### P1-R2：Geometry eligibility

继续采用已批准的 duplicate-plane policy：一个 exact duplicate 在派生 volume 中确定性保留一张；四个 same-plane different-content series 整 series 排除。

##### 验收标准

- 原始 DICOM 不修改。
- 后续 pipeline 只使用 1,014 eligible CT series。
- 四个排除 series 的脱敏 keys 保持 P1 永久记录。

### Phase 2：V2 physical-nodule cohort 语义

#### P2-R1：Primary class 与 stable physical units

Physical nodules 和 stable `nodule_uid` 完全复用已验收 P2 clustering/provenance。Primary inclusion 继续由 XML `nodule >=3 mm` class 决定；computed strict `>3 mm` 仅为 sensitivity flag。

##### 验收标准

- 不重新聚类、不改变任何 stable `nodule_uid`。
- P1 四个 excluded series 继续排除。
- 2,651/875 仅为 reference reconciliation，不是 hard gate。

#### P2-R2：V2 task eligibility

将所有 required targets 有效的 clusters 标记为 primary regression；只将 extremes 标记为 secondary eligible。

##### 验收标准

- 2,633 primary / 868 patients。
- 1,073 extreme / 578 patients，782 low、291 high。
- 1,560 middle-spectrum nodules 保留在 primary regression。
- 1 个 missing-required-target cluster 保留审计但不进入训练。

#### P2-R3：Stable provenance 与 reader aggregation

保留 source-derived fingerprints、diagnostic-only SQL IDs、9 个 valid-reader counts、raw ratings、dispersion、ties 和 strict diameter。

##### 验收标准

- V2 manifest 的 `nodule_uid` 集合与 V1 physical manifest 完全一致。
- SQL ID 不参与 V2 identity。
- 所有 target-ready rows 的 normalized malignancy target 位于 `[0,1]`。

### V2M：Baseline-v2 Protocol Migration

#### V2M-R1：协议与治理切换

创建 V2 requirements、protocol index，并让状态文档和 AGENTS 动态指向 active protocol。

##### 验收标准

- 协议索引只有一个 `ACTIVE` 版本。
- V1 标记 `SUPERSEDED/audit-only` 且内容无 diff。
- P3 保持 `NOT_STARTED`。

#### V2M-R2：配置冻结

生成 V2 source/resolved config 和 SHA-256。

##### 验收标准

- 重复 freeze 产生相同 hash。
- Resolved config 为只读文件。
- V1 config/hash 不变。

#### V2M-R3：Linear-regression smoke

完成 CPU/MPS/CUDA forward/backward，并保存 versioned audit。

##### 验收标准

- 三设备均 `PASS`。
- 输出直接来自 DenseNet final linear layer。
- MSE 直接比较未裁剪 output 与 normalized target。

#### V2M-R4：Cohort rematerialization

从本地私有 P2 manifest 确定性生成 V2 manifest 与脱敏 audit，不读取或修改原始 DICOM/XML。

##### 验收标准

- Counts 精确匹配 P2-R2。
- Stable `nodule_uid` 不变且唯一。
- Tracked audit 不含 patient ID、UID 或绝对路径。

#### V2M-R5：迁移阶段门

完整测试和双 agent 审查通过后进入 `AWAITING_USER_APPROVAL`。

##### 验收标准

- V1 immutable check、config tests、smoke tests、cohort tests 全部通过。
- 用户确认前不推送，不进入 P3。

### Phase 3：Consensus mask 与 ROI

#### P3-R1：Consensus mask

对每个 primary regression nodule 的 reader segmentations 使用 50% consensus threshold。

##### 验收标准

- Mask 非空、与 CT 对齐并可追溯到 source annotations。
- 空 mask 或异常 mask 被记录并阻止进入训练。

#### P3-R2：固定 crop/resize

Consensus non-zero tight bbox → image/mask 同步 crop → cube padding → `64³` resize。奇数 padding 额外 voxel 放高索引侧；image padding `-1000 HU`；image trilinear；mask nearest；clip `[-1000,700]` 后归一化 `[0,1]`。

##### 验收标准

- Image/mask shape 均为 `[1,64,64,64]`。
- Mask 只有 0/1，原 mask 全包含在 crop。
- `-1000 HU` 精确映射为 0。
- 保存 bbox、padding 和 transform metadata；重复处理字节一致。

#### P3-R3：ROI QA

##### 验收标准

- 随机、最小、最大和 reader-count 分层样本生成三平面 QA。
- 报告 volume、bbox、padding 分布并完成人工对齐确认。

### Phase 4：Patient-level splits 与共享初始化

#### P4-R1：Outer/inner split

Patient-grouped 5-fold outer CV；outer development/test 约 80/20；development 中约 12.5% patients 为 validation，整体约 70/10/20。

使用固定五级 strata：

```text
mean <= 2
2 < mean < 3
mean = 3
3 < mean < 4
mean >= 4
```

##### 验收标准

- Train/validation/test patients 互斥。
- 每位 patient 恰好作为 outer-test 一次。
- 每个 validation/test fold 同时包含 low/high extremes。
- 四模型共用同一 split files。

#### P4-R2：Train-only statistics

Contribution centering 和任何训练统计只能由当前 fold train subset 计算。

##### 验收标准

- Validation/test 不参与统计。
- 自动 patient leakage test 通过。

#### P4-R3：Shared encoder initialization

每 fold 只创建一次未训练 DenseNet-121 encoder state，四模型加载完全相同 state。

##### 验收标准

- 每折一个 initialization artifact/hash。
- 四模型训练前 encoder hashes 完全一致，否则阻断。
- Checkpoint 保存 initialization hash。

### Phase 5：Black-box DenseNet regression

#### P5-R1：模型与 loss

MONAI 3D DenseNet-121 输入单通道 `64³`，输出一个 unconstrained linear `raw_task_score`。

```text
L_blackbox = MSE(raw_task_score, target_normalized)
```

##### 验收标准

- Output shape `[B]` 或 `[B,1]`。
- 不读取 concepts，不产生 concept artifacts。
- 小数据 overfit 能降低 MSE。
- 不存在 output activation/clipping。

#### P5-R2：Checkpoint

最多 80 epochs；选择最小 validation `L_task`，相同值选更早 epoch。

##### 验收标准

- Metadata 保存 objective、epoch、config 和 encoder hash。
- Test 不参与选择，只在 checkpoint 固定后运行一次。

### Phase 6：Standard CBM regression

#### P6-R1：Concept predictor

输出 6 个 sigmoid continuous concepts、4-class internalStructure softmax 和 6-class calcification softmax。

```text
L_concept = (1/8) * sum(L_j)
```

##### 验收标准

- 八组 loss 等权；categorical 使用完整 vote distribution。
- 每组 loss 单独记录。

#### P6-R2：Sequential task training

先按最小 validation `L_concept` 选择 concept predictor；冻结 encoder/heads/BatchNorm 后，使用 frozen predicted concept vector 训练 unconstrained linear task head，目标为 MSE。

##### 验收标准

- Task stage 不改变 concept predictor state。
- Task head 不读取 DenseNet feature。
- 选择最小 validation MSE，相同选更早 epoch。

#### P6-R3：Contributions

Continuous group 为 scalar×weight；categorical group 为 distribution 与组 weights 内积。

##### 验收标准

- Bias 加八组 contribution 在 `1e-6` 内重建 `raw_task_score`。
- Rating-point conversion 在 `1e-6` 内重建 `score_1_to_5`。

### Phase 7：Mixed-type CEM regression

#### P7-R1：扩展声明

方法固定称为：

> A project-specific mixed-type extension of the original CEM.

##### 验收标准

- README、方法和结果均声明 mixed-type 是项目扩展。

#### P7-R2：Sample-conditioned states

所有 continuous/categorical state embeddings 必须由当前 sample 的 `h(x)` 动态生成；每组 embedding size 16。禁止全局静态 state table。

##### 验收标准

- 固定 concept probabilities、改变 `h(x)` 时 states 改变。
- Batch 内每个样本使用自身 states。
- Task head 只读取八组 embeddings。

#### P7-R3：Loss 与 training intervention

```text
L_CEM = MSE(raw_task_score, target_normalized) + 0.01 * L_concept
```

每组以 `p=0.25` 独立替换 mixture weights 为 ground truth target，不替换 sample-conditioned states。

##### 验收标准

- 记录 MSE、concept loss、total loss 和 intervention rate。
- 固定 seed 重现 masks；长期比例接近 25%。
- 最小 validation total loss 选 checkpoint。

#### P7-R4：Contributions

每组 contribution 为 16 维 embedding block 与对应 linear task weights 的内积。

##### 验收标准

- 原始和 intervention 后均在 `1e-6` 内重建 raw/rating scores。

### Phase 8：CBM + GAM regression

#### P8-R1：Additive model

```text
raw_task_score = bias + sum(group_contribution_j)
L_GAM = MSE(raw_task_score, target_normalized) + L_concept
```

##### 验收标准

- Task model 只读取 predicted concepts。
- Total loss 精确等于两项之和。
- 八组加 bias 精确重建 raw score。

#### P8-R2：Learned-softmax subnetworks

每组 5 个独立 subnetworks；continuous `1→32→16→1`，categorical `K→32→16→1`，ReLU hidden、linear scalar output。

```text
alpha_j = softmax(a_j)
group_contribution_j = sum_s alpha_j,s * f_j,s(concept_j)
```

`S=5`、`32→16`、learned-softmax 是项目预注册选择，最初冻结于 V1 并原样延续到 V2，不是 reference-paper hyperparameters。

##### 验收标准

- 每组恰有 5 个 subnetworks，alpha 非负且和为 1。
- `a_j` 获得 gradient 并保存 checkpoint。
- 禁止简单平均和跨 concept 输入。

#### P8-R3：Checkpoint

最多 80 epochs；最小 validation `L_GAM`，相同选更早 epoch。

##### 验收标准

- Metadata 保存 validation MSE、concept loss、total loss 和 alpha。
- Test 不参与选择。

### Phase 9：统一评估、干预与解释

#### P9-R1：Primary regression metrics

Primary metric：pooled OOF MAE，单位为原始 1–5 rating points。辅助报告 normalized MAE、original-scale RMSE、Pearson、Spearman 和五折结果。

##### 验收标准

- 每个 primary nodule 恰有一个 OOF prediction，四模型集合相同。
- 所有指标使用未裁剪 score。
- 报告 normalized/original prediction min、max 和上下界越界率。

#### P9-R2：Secondary extreme metrics

同一连续 score 在 extreme subset 报告 pooled OOF AUROC/AUPRC。Threshold-dependent metrics 使用每 fold validation-only Youden-J；tie 选择数值最大 threshold。Normalized `0.5` 仅为 sensitivity。

##### 验收标准

- 不建立 binary head，不称为 probability。
- 不报告 Brier/ECE，不进行 calibration。
- Test labels 不参与 threshold selection。
- `k=0` secondary intervention 精确重现 pooled OOF AUROC。

#### P9-R3：Concept metrics

Continuous：MAE、RMSE、Pearson、Spearman。Categorical：soft cross-entropy、multiclass Brier、hard modal macro-F1。

##### 验收标准

- 每项报告实际 N；soft 包含 ties，hard 排除 true ties。

#### P9-R4：Contribution centering 与单位

```text
mu_j = mean_train(r_j)
c_j = r_j - mu_j
b_centered = b + sum(mu_j)
```

Normalized reconstruction：`b_centered + sum(c_j) = raw_task_score`。

Rating-scale conversion：`rating_contribution_j=4*c_j`，`rating_bias=1+4*b_centered`。

##### 验收标准

- Means 只来自 train subset。
- 两种量纲 reconstruction 最大误差均不超过 `1e-6`。
- 报告明确标注 contribution units。

#### P9-R5：Primary intervention MAE curve

在每个 `k=0..8`、每个固定 permutation 上拼接五折 OOF predictions 后计算 MAE：

```text
x_k = k / 8
E_k = mean_over_100_permutations(pooled_OOF_MAE_at_k)
iMAE = trapezoidal_integral(E_k over x)
Delta_iMAE = E_0 - iMAE
```

Error-first 顺序仍按每个 sample 的 concept prediction error，不使用 malignancy target 排序。

##### 验收标准

- `k=0` 精确复现 primary pooled OOF MAE。
- 正 `Delta_iMAE` 表示改善。
- Fold mean±SD 明确标为 secondary。

#### P9-R6：Secondary intervention AUROC curve

在 extreme subset 对同一 intervention predictions 计算 pooled OOF AUROC、iAUC 与 `Delta_iAUC`。

##### 验收标准

- `k=0` 精确复现 secondary pooled OOF AUROC。
- Random permutations 跨模型/折可复现。

#### P9-R7：Grad-CAM

DenseNet 最后 convolutional feature layer 为 target layer。Malignancy target 为 `raw_task_score`；continuous concept target 为 pre-sigmoid concept logit；categorical 为 predicted-class logit。

##### 验收标准

- Black-box 每样本 1 张 map；concept models 每样本 9 张。
- Maps 上采样至 `64³`；全零标记 undefined。
- 空间指标仅称为 nodule-level proxy。

#### P9-R8：Occlusion faithfulness

选择确定性的最高 26,215 heatmap voxels并置 normalized image 为 0；random baseline 使用 20 个等大小、完整 ROI 均匀无放回 masks。

```text
Delta_saliency_abs = abs(original_raw_task_score - occluded_raw_task_score)
```

##### 验收标准

- Saliency/random voxel 数一致，tie 按较小 flat index。
- Random masks 可由 seed 完全重建。
- 报告 `Delta_saliency_abs - mean(Delta_random_abs)` 和胜过 random mean 的比例。
- 原始 image 不原地修改。

#### P9-R9：统计推断

Primary pooled OOF metrics 使用 2,000 patient-cluster bootstrap percentile 95% CI。Between-model 使用共享 patient draws 的 paired bootstrap。

##### 验收标准

- Primary regression 报告 paired `Delta_MAE`。
- Secondary extreme 报告 paired `Delta_AUROC`；单类 replicate 重抽直到 2,000 个有效 replicates。
- Bootstrap unit 是 patient，被抽中 patient 携带其全部 nodules。
- 四模型 OOF sets 不一致时禁止 paired comparison。

### Phase 10：Katana 正式实验与报告

#### P10-R1：OpenPBS jobs

每 model/fold 独立 CUDA job，只上传代码、V2 ROI、manifest 和 splits，不上传原始约 125 GB DICOM。

##### 验收标准

- Job 声明 GPU/CPU/RAM/walltime/path，可从 checkpoint 恢复。
- 正式结果记录 V2 config、split 和 encoder hashes。
- 每 fold 四模型初始 encoder hashes 相同。

#### P10-R2：最终报告

报告 cohort flow、regression/extreme performance、2,000 patient bootstrap、paired differences、concept quality、ties、intervention、centered contributions、GAM alphas、Grad-CAM、occlusion、定性/失败案例和限制。

##### 验收标准

- 表格可由 OOF artifacts 重建。
- 明确 primary score 未裁剪并报告越界率。
- 不误写为 pathology-confirmed diagnosis。
- Mixed-type CEM 与预注册 GAM choices 正确标记。
- 明确不是临床诊断系统。

## 5. 全局自动验收测试

正式实验前必须通过：

1. XML–DICOM UID mapping
2. Physical nodule clustering determinism
3. Stable source-derived nodule UID 与 SQL ID independence
4. Per-target valid-reader counts
5. V2 target normalization 与 extreme boundaries
6. Unconstrained linear output/no-clipping
7. Patient leakage 与 five-stratum split
8. Consensus mask alignment
9. Trilinear image/nearest mask resize
10. 八组等权 `L_concept`
11. Shared encoder initialization hash
12. 四模型 checkpoint selection
13. CEM dynamic sample-conditioned embeddings
14. GAM learned-softmax weights
15. CBM/CEM/GAM normalized contribution reconstruction
16. Rating-point contribution reconstruction
17. Train-fold centering
18. Validation-only Youden-J
19. Regression intervention `k=0` pooled OOF MAE
20. Extreme intervention `k=0` pooled OOF AUROC
21. iMAE/iAUC trapezoid calculation
22. 2,000 patient-cluster bootstrap 与 paired bootstrap
23. Deterministic 26,215-voxel occlusion
24. Black-box output-scope restriction
25. CPU/MPS/CUDA regression smoke
26. OOF sample-set equality

阻断条件：

- 任一 patient leakage
- 任一 shared encoder hash mismatch
- Regression output 被 sigmoid/tanh/clamp
- 任一 contribution reconstruction error `>1e-6`
- Test data 参与 checkpoint 或 threshold selection
- CEM 使用静态 concept state table
- GAM 使用简单平均
- 四模型 OOF sample sets 不一致

Reference 2,651/875 不匹配本身不构成失败。

## 6. 固定非目标

Baseline-v2 不包含：

- 病理确诊标签训练
- Patient Diagnoses XLS 主模型训练
- 临床变量或 radiomics
- Segmentation model 训练
- 外部数据集验证
- 多 seed variance
- 独立 binary classification head
- Test-time threshold tuning
- Post-hoc probability calibration
- Brier/ECE malignancy probability metrics
- Concept-region ground-truth localization claims

新增 sensitivity experiment 必须与 Baseline-v2 主结果分开报告。
