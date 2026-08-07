# LIDC-IDRI Baseline-v1 需求文档

**状态：已确认并冻结**  
**冻结日期：2026-08-08**

本需求文档冻结后，实施期间不得擅自修改科学协议。任何变更必须得到明确批准，并创建新的协议版本，不能覆盖 Baseline-v1。

## 1. 项目目标

在完全相同的数据 cohort、patient-level splits、3D ROI、DenseNet-121 encoder architecture 和 fold-specific encoder initialization 下，比较：

1. Black-box 3D DenseNet-121
2. Standard CBM
3. Mixed-type CEM
4. CBM + GAM

任务固定命名为：

> Radiologist-assessed pulmonary nodule malignancy classification

预测目标来自 LIDC-IDRI 放射科医生 malignancy ratings，不得表述为病理确诊癌症分类。项目不是临床诊断系统。

核心比较维度：

- 恶性度预测性能
- 概念预测质量
- 概念干预有效性
- 8 组概念贡献
- 结节级空间可解释性

数据限制以 [TCIA LIDC-IDRI 官方说明](https://www.cancerimagingarchive.net/collection/lidc-idri/)为准。

## 2. 全局实验协议

### G-R1：实验单位

一个样本对应一个 physical pulmonary nodule，而不是一条 reader annotation。

同一结节的多个 reader annotations 必须先通过 pylidc clustering 聚合，再生成：

- malignancy target
- 8 组 concept targets
- 50% consensus mask
- 3D ROI

#### 验收标准

- 每个 `nodule_uid` 只对应一个 CT series 和一个 physical nodule。
- 每个 reader annotation 可追溯到 canonical XML source。
- 同一 patient 的全部 CT、时间点和 nodules 始终属于同一数据划分。
- 不存在 annotation-level 或 patient-level leakage。

### G-R2：模型输出范围

#### Black-box 输出

Black-box 只允许输出：

```text
malignancy_logit
malignancy_probability
malignancy Grad-CAM
```

Black-box 不输出：

- concept predictions
- concept contributions
- concept-specific Grad-CAMs
- concept interventions

#### Standard CBM、CEM、CBM+GAM 输出

三个 concept models 输出：

```text
malignancy_logit
malignancy_probability
8-group concept predictions
8 raw group contributions
8 centered group contributions
centered bias
malignancy Grad-CAM
8 concept-specific Grad-CAMs
```

#### 验收标准

- Black-box 结果文件中不存在伪造或空占位 concept outputs。
- Concept prediction、contribution 和 concept-specific Grad-CAM 只适用于 CBM/CEM/GAM。
- 每项输出均可通过 `nodule_uid`、fold、model 和 checkpoint 唯一追溯。
- CBM/CEM/GAM 不允许 DenseNet feature 直接绕过概念层进入 malignancy predictor。

### G-R3：Malignancy target

对同一 physical nodule 的有效 malignancy ratings 取均值：

- `mean_malignancy <= 2`：benign，标签 0
- `mean_malignancy >= 4`：malignant，标签 1
- `2 < mean_malignancy < 4`：uncertain，不进入主要实验

Patient Diagnoses XLS：

- 不进入 Baseline-v1 训练。
- 不得覆盖 XML reader ratings。
- 只可在未来 supplementary analysis 使用。

#### 验收标准

- Manifest 保存原始 ratings、有效 reader 数、均值、标签和排除原因。
- uncertain nodules 不出现在主要 train/validation/test 中。
- 报告 benign、malignant、uncertain 的逐阶段数量。

### G-R4：概念定义

固定使用以下 8 个 concept groups：

1. subtlety
2. internalStructure
3. calcification
4. sphericity
5. margin
6. lobulation
7. spiculation
8. texture

其中：

- 连续/序数概念：subtlety、sphericity、margin、lobulation、spiculation、texture
- internalStructure：4-class categorical distribution
- calcification：6-class categorical distribution
- malignancy 是 target，不得作为输入概念

连续评分按合法范围线性归一化至 `[0,1]`，再计算 reader mean。

Categorical concepts 使用完整 reader vote distribution 作为 soft target。

#### 验收标准

- 6 个连续 target 均位于 `[0,1]`。
- categorical target probability sum 在数值误差内等于 1。
- 原始评分、归一化评分、聚合评分均可审计。
- 模型 concept input schema 中不存在 malignancy。

### G-R5：协议冻结与变更控制

本需求文档已冻结为 Baseline-v1。

#### 验收标准

- 机器可读配置保存 protocol version 和配置 SHA-256。
- 正式 split、checkpoint、prediction 和 report 均记录配置哈希。
- 实施期间不得静默修改 cohort、loss、model、threshold、metrics 或统计定义。
- 任何科学协议变更必须经明确批准，并创建新版本，不能覆盖 Baseline-v1。

## 3. 标准接口与产物

目标代码仓库：

```text
/Users/katherine/Desktop/lidc_baseline
```

原始数据：

```text
/Users/katherine/Desktop/lidc_data
```

标准产物：

```text
configs/baseline_v1.yaml
artifacts/audit/
artifacts/manifests/nodules.parquet
artifacts/splits/fold_{0..4}.json
artifacts/encoder_initializations/fold_{0..4}.pt
artifacts/rois/{nodule_uid}.npz
runs/{model}/fold_{k}/checkpoint.pt
runs/{model}/fold_{k}/predictions.parquet
runs/{model}/fold_{k}/contributions.parquet
runs/{model}/fold_{k}/metrics.json
reports/baseline_v1/
```

标准 ROI：

```text
image: float32 [1,64,64,64]
mask:  uint8   [1,64,64,64]
```

`nodules.parquet` 至少保存：

- patient/study/series identifiers
- stable physical `nodule_uid`
- stable source annotation identifiers/fingerprints
- diagnostic-only pylidc SQL IDs
- reader annotation fingerprints
- 每个 concept 的有效 reader 数
- malignancy 原始 ratings、均值、标签
- 8 组原始及聚合 concept targets
- categorical tie flags
- reader count 和 `has_at_least_3_readers`
- consensus/crop/padding/resize metadata
- computed diameter 和 strict `>3 mm` flag
- exclusion/status flags
- source XML/DICOM fingerprints

### 验收标准

- 所有训练和评估脚本只读取标准 manifest、split 和 ROI。
- 删除派生产物后，可通过 source fingerprints 重建。
- 产物不依赖临时目录。
- 每个 checkpoint 保存 fold、seed、配置哈希、encoder initialization hash、objective 和最佳 epoch。

## 4. 分阶段需求

### Phase 0：工程环境与配置冻结

#### P0-R1：运行环境

技术栈固定为：

- Python
- PyTorch
- MONAI
- pylidc
- pydicom
- pandas/pyarrow
- scikit-learn/scipy

Mac CPU/MPS 用于开发和 smoke tests；UNSW Katana CUDA 用于正式训练。

##### 验收标准

- CPU、MPS、CUDA 均可完成数据加载和 forward/backward smoke test。
- 保存 Python、PyTorch、MONAI、pylidc 和 CUDA 版本。
- 所有随机过程使用固定 seed。
- 每个 fold 只运行一个正式 seed。

#### P0-R2：冻结配置

配置必须记录：

- cohort 和标签协议
- concept schema
- ROI preprocessing
- split
- shared encoder initialization
- model structures
- losses
- checkpoint rules
- threshold rules
- intervention definitions
- bootstrap definitions
- Grad-CAM 和 occlusion definitions

##### 验收标准

- 生成只读 resolved config 和 SHA-256。
- 配置哈希在正式 split 生成前冻结。
- 后续结果记录相同配置哈希。

**阶段门：** 环境 smoke test 和配置审计通过后进入 Phase 1。

### Phase 1：DICOM/XML 审计

#### P1-R1：Canonical XML source

`LIDC-XML-only` 是 canonical annotation source。

DICOM 下载目录中的 XML 只用于交叉核对，不能混合产生重复 annotations。

当前本地参考盘点：

- 1,010 patient directories
- 1,018 CT series
- 243,958 CT DICOM
- 513 DX
- 56 CR
- 1,319 XML，包含 CT 和 CXR XML
- canonical XML 覆盖 1,018 CT series

##### 验收标准

- 分别统计 CT、DX、CR 和 CXR。
- 每个 canonical CT XML 映射至唯一 CT series。
- DX、CR 和 CXR XML 不进入 cohort。
- 重复、缺失、无法解析和 UID 冲突均有异常清单。

#### P1-R2：DICOM geometry

验证：

- Study/Series/SOP Instance UID
- ImagePositionPatient
- ImageOrientationPatient
- PixelSpacing
- SliceThickness
- InstanceNumber
- duplicate/missing slices

切片必须按空间坐标投影排序，不得只按文件名或 InstanceNumber 排序。

##### 验收标准

- 每个纳入 CT series 能确定性构建 3D volume。
- 方向、spacing、重复切片和异常 gap 均被检测。
- XML、CT series、volume 三者映射唯一。

**阶段门：** 数据审计报告通过后进入 Phase 2。

### Phase 2：Physical nodule cohort

#### P2-R1：Primary cohort

Primary cohort 由 XML 中 LIDC `nodule ≥3 mm` annotation class 决定。

排除：

- `<3 mm nodule`
- `non-nodule`

论文报告的 **2,651 nodules / 875 patients** 只作为 reference-paper reconciliation benchmark，不是 hard equality gate。参考 [Dumaev et al.](https://arxiv.org/abs/2405.17483)。

Computed strict `>3 mm` 仅作为 sensitivity flag，不决定 primary inclusion。

##### 验收标准

- Primary inclusion 不依赖 computed diameter。
- 输出 annotation-to-physical-nodule 映射。
- 输出 class-selected、binary-eligible 和 reference count reconciliation。
- 与 2,651/875 不一致不构成失败。
- 不允许为了匹配论文数量修改 cohort。

#### P2-R2：Stable provenance 与 nodule UID

Canonical `nodule_uid` 不得只由 pylidc SQL integer ID 生成。

Stable provenance 至少使用：

- patient/study/series UID
- canonical XML SHA-256
- XML reading-session/nodule identifier（若可获得）
- SOP/contour-derived fingerprint
- characteristic/source annotation fingerprint
- physical cluster 内排序后的 annotation fingerprints

pylidc SQL IDs 只可作为 diagnostic lookup field。

##### 验收标准

- 更换或重建 pylidc database 后，只要原始 XML/DICOM 未变，stable `nodule_uid` 保持不变。
- 修改 SQL integer ID 不得改变 canonical `nodule_uid`。
- 修改实际 source annotation 内容必须改变对应 fingerprint。
- 每个 `nodule_uid` 唯一，无碰撞。

#### P2-R3：Reader aggregation

不要求至少 3 位 reader。只要求 malignancy 和 8 个 concepts 各至少存在一个有效 rating。

每个 nodule 保存：

- overall reader count
- `has_at_least_3_readers`
- 每个 concept 的 valid-reader count
- malignancy valid-reader count
- reader 间离散程度
- annotation source fingerprints

##### 验收标准

- Manifest 包含 9 个逐目标 valid-reader counts。
- 缺失目标导致的排除必须记录具体字段。
- 可直接生成 `>=3 readers` sensitivity cohort，无需重新解析 XML。

#### P2-R4：Categorical ties

Modal vote tie 样本：

- 保留用于训练。
- 保留用于 soft metrics。
- 从 hard modal-label macro-F1 中排除。
- 报告每个 concept、fold 和 pooled OOF 的 tie 数量。

##### 验收标准

- Soft metric sample count 包含 ties。
- Hard macro-F1 sample count只包含 unique true modal label。
- Tie 排除不改变主要 cohort。

**阶段门：** Cohort、provenance 和 reconciliation 审计通过后进入 Phase 3。

### Phase 3：Consensus mask 与 ROI

#### P3-R1：Consensus

同一 physical nodule 的 reader segmentations 使用 50% consensus threshold。

##### 验收标准

- Consensus mask 非空。
- Mask 可追溯到参与 consensus 的 annotations。
- Mask 与 CT volume 正确配准。
- 空 mask 和异常 mask 被阻止进入训练。

#### P3-R2：Baseline-v1 ROI 算法

该流程属于 Baseline-v1 implementation choice：

1. Consensus 后计算 non-zero mask tight 3D bbox。
2. 同步裁剪 image 和 mask。
3. Padding 成 cube，边长取三个 bbox 维度的最大值。
4. 奇数 padding 的额外 voxel 放在高索引侧。
5. Image padding 使用 `-1000 HU`。
6. Mask padding 使用 0。
7. Image 使用 3D trilinear resize 到 `64³`。
8. Mask 使用 nearest-neighbour resize 到 `64³`。
9. Image clip 至 `[-1000,700] HU`。
10. 线性归一化至 `[0,1]`。

##### 验收标准

- Image/mask shape 均为 `[1,64,64,64]`。
- Mask resize 后只有 0/1。
- 原 consensus mask 全部包含在 crop 内。
- `-1000 HU` 精确映射为 0。
- 保存 bbox、padding、interpolation 和坐标 metadata。
- 同一样本重复处理结果完全相同。

#### P3-R3：ROI QA

##### 验收标准

- 随机、最小、最大和不同 reader-count 样本均生成三平面 QA 图。
- 报告 mask volume、bbox 和 padding 比例分布。
- 人工确认不存在明显错位、空 ROI 或非目标区域。

**阶段门：** ROI 自动测试和人工 QA 通过后进入 Phase 4。

### Phase 4：Patient-level split 与共享初始化

#### P4-R1：Outer split

固定 patient-grouped 5-fold outer cross-validation：

- 约 80% outer development
- 约 20% outer test

##### 验收标准

- Development/test patient 不重叠。
- 每个 patient 恰好作为 outer-test patient 一次。
- 每个 test fold 同时包含 benign 和 malignant。
- 报告 patient/nodule/class 分布。

#### P4-R2：Validation split

从 outer-development patients 中选择约 12.5% 作为 validation，得到总体约：

- 70% train
- 10% validation
- 20% test

##### 验收标准

- Train/validation/test patient 三者互斥。
- Validation 约为 outer-development patients 的 12.5%。
- 报告实际 patient-level 和 nodule-level 比例。
- 四个模型共用同一 split manifest。

#### P4-R3：Train-only statistics

只能使用当前 fold train subset 计算：

- `pos_weight`
- contribution centering means
- 其他训练统计

```text
pos_weight = N_negative / N_positive
```

##### 验收标准

- Validation/test 不参与统计计算。
- 每个 fold 保存 train-only statistics。
- 自动 leakage test 必须通过。

#### P4-R4：四模型共享 encoder initialization

每个 fold 必须：

1. 使用该 fold seed 创建一次未训练的 MONAI 3D DenseNet-121。
2. 立即保存冻结的 encoder initial state。
3. 计算该 state 的稳定 SHA-256。
4. Black-box、Standard CBM、CEM、GAM 均加载这一份完全相同的 encoder state。
5. 各模型的 concept/task heads 可以按各自结构独立初始化。

##### 验收标准

- 每折存在唯一 encoder initialization artifact。
- 四模型训练前 encoder state hashes 完全一致。
- 任一 hash 不一致视为阻断性失败。
- Checkpoint 保存 initial encoder hash。
- Test data 不参与 initialization。

**阶段门：** Split、leakage 和 encoder hash 检查通过后进入 Phase 5。

### Phase 5：Black-box DenseNet

#### P5-R1：模型

使用 MONAI 3D DenseNet-121，从头训练，输入单通道 `64³` ROI，输出一个 malignancy logit。

##### 验收标准

- Output shape 为 `[B]` 或 `[B,1]`。
- 不读取 concept targets。
- 不产生 concept outputs。
- 小规模 overfit/sanity test 能降低训练 loss。

#### P5-R2：Loss 与 checkpoint

```text
L_blackbox = weighted_BCE
```

最多 80 epochs。

Checkpoint：

> 最小 validation weighted_BCE；完全相同时选择更早 epoch。

##### 验收标准

- Checkpoint metadata 保存最佳 epoch 和 validation objective。
- Test set 不参与选择。
- 固定 checkpoint 后才运行 test inference。

### Phase 6：Standard CBM

#### P6-R1：Concept predictor

输出：

- 6 个 sigmoid continuous predictions
- 4-class internalStructure softmax
- 6-class calcification softmax

显式 concept vector 总维度为 16。

```text
L_concept = (1/8) × Σ(j=1..8) L_j
```

其中：

- 连续组：normalized MSE
- categorical 组：soft-target cross-entropy

##### 验收标准

- 8 组 loss 分别记录。
- `L_concept` 等于 8 组 loss 的算术平均。
- Categorical loss 直接使用 vote distribution。

#### P6-R2：Sequential training

1. 使用 `L_concept` 训练 concept predictor。
2. 选择最小 validation `L_concept` checkpoint。
3. 冻结 encoder 和 concept heads。
4. 使用 frozen predicted concept vector 训练 linear logistic task head。
5. 选择最小 validation weighted BCE checkpoint。

每阶段最多 80 epochs；相同 objective 选择更早 epoch。

##### 验收标准

- Task stage 中 concept predictor 参数及 BatchNorm 状态不变。
- Task head 不访问 DenseNet feature。
- Checkpoint selection 不引用 test set。

#### P6-R3：Raw contributions

- Continuous group：scalar × corresponding linear weight
- Categorical group：probability vector 与本组 weights 的内积

##### 验收标准

```text
logit = bias + Σ raw_contribution_j
```

最大 reconstruction error 不超过 `1e-6`。

### Phase 7：Mixed-type CEM

#### P7-R1：扩展声明

本项目方法必须称为：

> A project-specific mixed-type extension of the original CEM.

参考 [CEM 论文](https://arxiv.org/abs/2209.09056)和[官方代码](https://github.com/mateoespinosa/cem)。

##### 验收标准

- README、方法和结果中都有扩展声明。
- 不声称 mixed-type 设计是原论文直接报告的 LIDC 实现。

#### P7-R2：Dynamic state embeddings

所有 state embeddings 由当前 sample 的 DenseNet feature `h(x)` 动态生成。

连续概念：

```text
e_j^0(x), e_j^1(x) = generator_j(h(x))
z_j(x) = (1-p_j(x))e_j^0(x) + p_j(x)e_j^1(x)
```

Categorical 概念：

```text
e_j^1(x),...,e_j^K(x) = generator_j(h(x))
z_j(x) = Σ_k p_jk(x)e_j^k(x)
```

每组 embedding size 固定为 16。

##### 验收标准

- 不存在作为 concept states 的全局静态 embedding table。
- 固定 concept probabilities、改变当前 sample feature 时，state embeddings 改变。
- Batch 内每个样本使用自身 feature。
- Task head 只读取 8 组 embeddings。

#### P7-R3：Loss 与 training intervention

```text
L_CEM = weighted_BCE + 0.01 × L_concept
```

训练时每个 concept group 独立以 `p=0.25` 被 ground truth 替换。

Checkpoint：

> 最小 validation `L_CEM`；相同值选择更早 epoch；最多 80 epochs。

##### 验收标准

- 分别记录 BCE、`L_concept`、总 loss 和 intervention rate。
- 固定 seed 可重现 intervention masks。
- Intervention 只改变 mixture weights，不替换 sample-conditioned states。
- Test set 不参与 checkpoint selection。

#### P7-R4：Contributions

每组 raw contribution 是该组 16 维 embedding 与 linear task weights 的内积。

##### 验收标准

- 8 组 contribution 加 bias 在 `1e-6` 内重建 logit。
- Concept intervention 后仍满足 reconstruction。

### Phase 8：CBM + GAM

#### P8-R1：Additive task model

```text
task_logit = bias + Σ group_contribution_j
L_GAM = weighted_BCE + L_concept
```

Task model 只读取 concepts。

##### 验收标准

- 不存在 DenseNet feature bypass。
- 总 loss 精确等于两项之和。
- 8 组 contribution 加 bias 重建 logit。

#### P8-R2：Learned-softmax GAM ensemble

每个 concept group 固定使用 5 个 concept-local subnetworks：

```text
continuous:  1 → 32 → 16 → 1
categorical: K → 32 → 16 → 1
```

Hidden activation 使用 ReLU。

每组具有可学习 logits：

```text
alpha_j = softmax(a_j)
group_contribution_j =
    Σ(s=1..5) alpha_j,s × f_j,s(concept_j)
```

以下全部属于 Baseline-v1 pre-registered choices，而不是 reference-paper reported hyperparameters：

- `S=5`
- `32→16`
- learned softmax weighting

##### 验收标准

- 每组恰有 5 个 subnetworks。
- `alpha_j,s >= 0` 且每组权重和为 1。
- `a_j` 参与梯度更新并保存在 checkpoint。
- 不再使用简单平均。
- 每个 subnetwork 只能读取其所属 concept group。
- 保存并报告每折最终 learned `alpha_j`。

#### P8-R3：Checkpoint

> 最小 validation `L_GAM`；相同值选择更早 epoch；最多 80 epochs。

##### 验收标准

- Checkpoint 保存 validation BCE、`L_concept`、`L_GAM` 和 learned weights。
- Test set 不参与选择。

### Phase 9：统一评估

#### P9-R1：预测指标与 validation-only threshold

Primary threshold-free metric：

- pooled OOF AUROC

其他 threshold-free metrics：

- AUPRC
- Brier score
- ECE

Threshold-dependent metrics：

- accuracy
- balanced accuracy
- sensitivity
- specificity
- macro-F1

每个 model/fold 的 threshold 必须只由该 fold validation predictions 选择：

```text
J(t) = TPR(t) - FPR(t)
t* = argmax J(t)
```

若多个 threshold 获得相同最大 Youden-J，选择数值最大的 threshold。

该 threshold 固定后应用于对应 outer-test fold。

`threshold=0.5` 只作为 secondary sensitivity analysis。

##### 验收标准

- AUROC/AUPRC 不依赖 threshold。
- Test labels 不参与 threshold selection。
- 保存每个 model/fold 的 validation Youden threshold。
- Pooled OOF threshold metrics 使用各样本所属 fold 的 validation threshold。
- 0.5 结果与 primary Youden-J 结果分开标记。

#### P9-R2：ECE

使用未经 post-hoc calibration 的 sigmoid probabilities。

固定 15 个 `[0,1]` equal-width bins：

```text
ECE = Σ_m |B_m|/N ×
      |mean_probability(B_m) - positive_rate(B_m)|
```

- 前 14 个 bin 左闭右开。
- 最后一个 bin 包含 1。
- Empty bins 忽略。

##### 验收标准

- 覆盖 `p=0`、boundary 和 `p=1` 单元测试。
- Primary ECE 使用 pooled OOF predictions。

#### P9-R3：Concept metrics

连续 concepts：

- MAE
- RMSE
- Pearson
- Spearman

Categorical concepts：

- soft cross-entropy
- multiclass Brier
- hard modal-label macro-F1

##### 验收标准

- 每项指标报告 sample count。
- Soft metrics 包含 ties。
- Hard macro-F1 排除 true modal ties。
- 报告各 concept/fold/OOF tie 数。

#### P9-R4：Contribution centering

适用于 Standard CBM、CEM、GAM，不适用于 Black-box。

```text
r_j(x) = raw contribution
μ_j = mean over current fold train subset of r_j(x)
c_j(x) = r_j(x) - μ_j
b' = b + Σ_j μ_j
```

因此：

```text
b' + Σ_j c_j(x) = original logit
```

##### 验收标准

- `μ_j` 只由 train subset 计算。
- 每个 model/fold 保存 8 个 means 和 centered bias。
- 最大 reconstruction error 不超过 `1e-6`。
- 主报告使用 centered contributions，并保存 raw values 审计。

#### P9-R5：Primary pooled-OOF intervention curve

只适用于 Standard CBM、CEM、GAM。

替换 selected concept groups：

- continuous：normalized reader mean
- categorical：reader vote distribution

```text
x_k = k/8, k=0,...,8
```

每个 fold 生成 100 个固定 permutations。对于 permutation index `r` 和 intervention level `k`：

1. 在每个 fold 使用该 fold 的第 `r` 个固定 permutation。
2. 对该 fold 全部 test samples 统一使用同一 group order。
3. 将五折 test predictions 拼接为 pooled OOF predictions。
4. 计算：

```text
A_r,k = pooled OOF AUROC
A_k = mean over r=1..100 of A_r,k
```

Primary：

```text
iAUC = trapezoidal integral of A_k over [0,1]
ΔiAUC = iAUC - A_0
```

因为 `k=0` 不发生 intervention：

```text
A_0 = primary pooled OOF AUROC
```

Fold mean ± SD 作为 secondary report。

Error-first upper-bound：

- continuous error：`|prediction-target|`
- categorical error：`0.5 × ||p-q||_1`
- 顺序逐样本计算
- 每个 `k` 拼接 pooled OOF predictions 后计算 AUROC

##### 验收标准

- `k=0` 精确复现主结果 pooled OOF AUROC。
- Primary curve 不使用 fold AUROC 简单平均代替 pooled OOF AUROC。
- 100 个 permutations 可复现并保存。
- Fold mean ± SD 明确标记为 secondary。
- Error-first 不使用 malignancy label 排序。

#### P9-R6：Grad-CAM scope

目标层固定为 DenseNet 最后一个 convolutional feature layer。

Black-box：

- 只生成 malignancy task-logit Grad-CAM。

CBM/CEM/GAM：

- malignancy task-logit Grad-CAM
- 6 个 continuous pre-sigmoid concept-logit Grad-CAMs
- 2 个 categorical predicted-class-logit Grad-CAMs

上采样至 `64³`。

空间 proxy metrics：

- positive heatmap energy inside consensus mask
- peak voxel inside mask
- occlusion faithfulness

##### 验收标准

- Black-box 每个样本只有 1 张 malignancy map。
- Concept models 每个样本有 1+8 张 maps。
- Heatmap 与 mask shape 相同。
- 全零 heatmap 标记 undefined 并报告数量。
- 不将指标描述为 concept-region localization accuracy。

#### P9-R7：Occlusion

Flatten `64³` heatmap，选择：

```text
ceil(0.10 × 64³) = 26,215 voxels
```

选择规则：

- Heatmap 从高到低。
- 相同数值时选择较小 flat index。
- 对应 normalized image voxels 设置为 0，即 `-1000 HU`。

```text
Δ_saliency =
original_target_logit - occluded_target_logit
```

Random baseline：

- 每个 sample/target 20 个 masks。
- 每个 mask 从完整 ROI 无放回均匀选择 26,215 voxels。
- Seed 由 baseline seed、fold、nodule UID 和 target 确定。

报告：

```text
Δ_saliency - mean(Δ_random)
proportion(Δ_saliency > mean(Δ_random))
```

##### 验收标准

- Saliency/random mask voxel 数一致。
- Random mask 不限制在 consensus mask 内。
- Random masks 逐 voxel 可重现。
- 原始 image tensor 不被原地修改。

#### P9-R8：统计推断

所有主要 pooled OOF performance estimates 使用：

- 2,000 patient-cluster bootstrap replicates
- 95% percentile confidence intervals
- 固定 bootstrap seed

每次 bootstrap：

1. 从 unique patients 中有放回抽样。
2. 被抽中的 patient 携带其全部 nodules。
3. 同一 patient 被抽中多次时，其全部 nodules 相应重复。
4. 若 replicate 只有一个 malignancy class，则重抽。
5. 直到得到 2,000 个有效 replicates。

Between-model AUROC differences 使用 paired patient-level bootstrap：

- 同一个 replicate 对所有模型使用完全相同的 patient draw。
- 模型间比较使用相同 OOF nodule set。
- 报告：

```text
ΔAUROC = AUROC_model_A - AUROC_model_B
95% bootstrap CI
```

##### 验收标准

- 每个主结果确实包含 2,000 个有效 replicates。
- Bootstrap unit 是 patient，不是 nodule。
- Paired comparison 使用共享 patient draws。
- 保存 bootstrap seed、replicate count 和 CI method。
- 四模型 OOF `nodule_uid` 集合不一致时禁止 paired comparison。

**阶段门：** OOF、threshold、concept、contribution、intervention、Grad-CAM、occlusion 和 bootstrap tests 全部通过后进入 Phase 10。

### Phase 10：Katana 正式实验与报告

#### P10-R1：Katana jobs

使用 OpenPBS/qsub 运行 CUDA jobs。

每个 model/fold 独立提交，使用预处理 ROI、manifest 和 split，不重复上传原始 125 GB DICOM。

##### 验收标准

- Job 明确 GPU、CPU、RAM、walltime 和路径。
- Interrupted job 可恢复。
- 正式结果使用冻结配置、split hash 和 encoder initialization hash。
- 每折四模型 initial encoder hashes 一致。

#### P10-R2：最终报告

报告至少包含：

- Cohort flow 和 reference reconciliation
- Stable provenance 和排除统计
- 五折划分统计
- 四模型 pooled OOF performance
- 2,000 patient-bootstrap 95% CIs
- Paired model AUROC differences
- Validation Youden thresholds
- Threshold 0.5 secondary sensitivity
- Concept quality 和 tie 数
- Pooled OOF intervention curves、iAUC、ΔiAUC
- Fold mean ± SD secondary intervention results
- 8 组 centered contributions
- GAM learned softmax weights
- Grad-CAM spatial proxy
- Occlusion faithfulness
- 定性案例、失败案例和限制

##### 验收标准

- 所有表格可由保存的 OOF predictions 重建。
- 不将 reader-assessed malignancy 写成 pathology-confirmed diagnosis。
- Mixed-type CEM 明确标记为项目扩展。
- GAM `S=5`、`32→16` 和 learned-softmax weighting 明确标记为 Baseline-v1 pre-registered choices。
- 明确声明不是临床诊断系统。

## 5. 全局自动验收测试

正式实验前必须通过：

1. XML–DICOM UID mapping
2. Physical nodule clustering determinism
3. Stable source-derived nodule UID
4. SQL ID independence
5. Per-concept valid-reader counts
6. Patient leakage
7. Consensus mask alignment
8. Trilinear image/nearest mask resize
9. Label threshold boundaries
10. Categorical vote ties
11. 八组等权 `L_concept`
12. 固定 checkpoint selection
13. 四模型 shared encoder initialization hash
14. Standard CBM logit reconstruction
15. CEM dynamic sample-conditioned embeddings
16. CEM logit reconstruction
17. GAM learned-softmax weights
18. GAM logit reconstruction
19. Train-fold contribution centering
20. Validation-only Youden-J
21. Pooled OOF intervention `k=0`
22. iAUC trapezoid calculation
23. 15-bin ECE boundaries
24. 2,000 patient-cluster bootstrap
25. Paired patient bootstrap
26. Deterministic 26,215-voxel occlusion
27. Black-box output-scope restriction
28. CPU/MPS/CUDA forward shape
29. OOF sample-set equality

阻断条件：

- 任一 patient leakage
- 任一 shared encoder hash mismatch
- Contribution reconstruction error `>1e-6`
- Test data 参与 checkpoint 或 threshold selection
- CEM 使用静态 concept state embedding table
- GAM 使用简单平均代替 learned softmax
- 四模型 OOF sample sets 不一致

Reference 2,651/875 不匹配不构成失败。

## 6. 固定非目标

Baseline-v1 不包含：

- 病理确诊标签训练
- Patient Diagnoses XLS 主模型训练
- 临床变量或 radiomics
- Segmentation model 训练
- 外部数据集验证
- 多 seed variance
- Test-time threshold tuning
- Post-hoc probability calibration
- Concept-region ground-truth localization claims

任何新增 sensitivity analysis 必须与 Baseline-v1 主结果分开报告。
