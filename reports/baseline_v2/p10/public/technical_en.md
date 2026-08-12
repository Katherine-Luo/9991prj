# Baseline-v2: Unified Evaluation of Interpretable 3D Models for LIDC-IDRI

**REPORT-DATA-SHA256:** `d9ae6d45308912b9dd6b23c26918bbcbb3790244d6a2575cd5b64d236d8d2807`

## 1. Executive Summary

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- 2,633 nodules; 868 patients; folds 479/502/539/549/564; patient leakage 0.
- Grad-CAM: 73,724 requested = 66,769 valid + 6,955 undefined post-ReLU zero maps.
- Bootstrap: 2,000 patient-cluster draws.
- Black-box: MAE 0.5006 [0.4829, 0.5195]; RMSE 0.6422; AUROC 0.9453; AUPRC 0.8937.
- Standard CBM: MAE 0.5021 [0.4828, 0.5223]; RMSE 0.6496; AUROC 0.9327; AUPRC 0.8657.
- Mixed-type CEM: MAE 0.4841 [0.4669, 0.5021]; RMSE 0.6283; AUROC 0.9417; AUPRC 0.8774.
- Learned-softmax GAM: MAE 0.4804 [0.4623, 0.4985]; RMSE 0.6176; AUROC 0.9493; AUPRC 0.9026.

## 2. Clinical and Scientific Context

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis [1].
- The system is a research benchmark and is not a clinical diagnostic product.
- The models used DenseNet-121 [2]; CBM and CEM terminology follows [3], [4], while Mixed-type CEM and Learned-softmax GAM denote the preregistered project-specific designs.

## 3. Cohort Construction

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- The canonical OOF cohort contained 2,633 unique nodules from 868 patients.
- Outer test fold counts were 479/502/539/549/564; patient leakage was 0.
- All 4 models used identical targets and fold-specific outer test membership.

## 4. Patient-grouped Five-fold Protocol

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- The canonical OOF cohort contained 2,633 unique nodules from 868 patients.
- Outer test fold counts were 479/502/539/549/564; patient leakage was 0.
- All 4 models used identical targets and fold-specific outer test membership.

## 5. Model Architectures

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- P5: Black-box DenseNet-121 regression [2].
- P6: Standard CBM with 8 activated concept groups [3].
- P7: project-specific Mixed-type CEM with 8 concept groups [4].
- P8: Learned-softmax GAM with 8 groups and 5 local experts per group.

## 6. Training and Test Governance

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- The execution registry contains 40 immutable model/fold records spanning P5-P9.
- P5-P8 each retained exactly 1 valid committed test evaluation per fold; P9 created 0 additional test evaluations.
- Verifier recoveries are distinguished from scientific execution and do not change persisted predictions or metrics.

## 7. Unified Evaluation Methods

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Primary scores were used without clipping; 2,000 shared patient-cluster bootstrap draws produced percentile 95% CIs [6].
- Secondary Youden-J thresholds used only fold-specific validation samples with malignancy <= 2 or >= 4.
- Each valid Grad-CAM map [5] used 26,215 saliency voxels and 20 matched random masks.
- Paired signs are DeltaMAE=MAE_A-MAE_B and DeltaAUROC=AUROC_B-AUROC_A; positive values favor model B.
- Intervention signs are Delta_iMAE=baseline_MAE-iMAE and Delta_iAUC=iAUC-baseline_AUROC; positive values denote improvement.

## 8. Primary Regression Results

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Black-box: MAE 0.5006 [95% CI 0.4829, 0.5195]; RMSE 0.6422 [95% CI 0.6189, 0.6671]; normalized MAE 0.1252 [95% CI 0.1207, 0.1299]; Pearson 0.7157 [95% CI 0.6894, 0.7409]; Spearman 0.6345 [95% CI 0.5994, 0.6676]; unclipped 1-5 prediction range [0.4885, 5.1200]; below 1 rate 0.0148; above 5 rate 0.0011.
- Standard CBM: MAE 0.5021 [95% CI 0.4828, 0.5223]; RMSE 0.6496 [95% CI 0.6254, 0.6749]; normalized MAE 0.1255 [95% CI 0.1207, 0.1306]; Pearson 0.7076 [95% CI 0.6770, 0.7354]; Spearman 0.6091 [95% CI 0.5700, 0.6482]; unclipped 1-5 prediction range [0.8585, 4.5798]; below 1 rate 0.0087; above 5 rate 0.0000.
- Mixed-type CEM: MAE 0.4841 [95% CI 0.4669, 0.5021]; RMSE 0.6283 [95% CI 0.6040, 0.6537]; normalized MAE 0.1210 [95% CI 0.1167, 0.1255]; Pearson 0.7296 [95% CI 0.7010, 0.7565]; Spearman 0.6400 [95% CI 0.6036, 0.6734]; unclipped 1-5 prediction range [0.8229, 4.9351]; below 1 rate 0.0042; above 5 rate 0.0000.
- Learned-softmax GAM: MAE 0.4804 [95% CI 0.4623, 0.4985]; RMSE 0.6176 [95% CI 0.5924, 0.6423]; normalized MAE 0.1201 [95% CI 0.1156, 0.1246]; Pearson 0.7405 [95% CI 0.7117, 0.7676]; Spearman 0.6528 [95% CI 0.6160, 0.6883]; unclipped 1-5 prediction range [1.0081, 4.6816]; below 1 rate 0.0000; above 5 rate 0.0000.
- DeltaMAE Black-box - Learned-softmax GAM = 0.0201 [95% CI 0.0101, 0.0305]; crosses zero: no.
- DeltaMAE Black-box - Mixed-type CEM = 0.0164 [95% CI 0.0060, 0.0272]; crosses zero: no.
- DeltaMAE Black-box - Standard CBM = -0.0016 [95% CI -0.0148, 0.0120]; crosses zero: yes.
- DeltaMAE Mixed-type CEM - Learned-softmax GAM = 0.0037 [95% CI -0.0059, 0.0130]; crosses zero: yes.
- DeltaMAE Standard CBM - Learned-softmax GAM = 0.0217 [95% CI 0.0109, 0.0326]; crosses zero: no.
- DeltaMAE Standard CBM - Mixed-type CEM = 0.0181 [95% CI 0.0059, 0.0301]; crosses zero: no.

## 9. Secondary Extreme-task Results

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Black-box: AUROC 0.9453 [95% CI 0.9264, 0.9615]; AUPRC 0.8937 [95% CI 0.8586, 0.9250].
- Standard CBM: AUROC 0.9327 [95% CI 0.9113, 0.9510]; AUPRC 0.8657 [95% CI 0.8263, 0.8995].
- Mixed-type CEM: AUROC 0.9417 [95% CI 0.9202, 0.9605]; AUPRC 0.8774 [95% CI 0.8332, 0.9155].
- Learned-softmax GAM: AUROC 0.9493 [95% CI 0.9266, 0.9685]; AUPRC 0.9026 [95% CI 0.8678, 0.9340].
- DeltaAUROC Learned-softmax GAM - Black-box = 0.0042 [95% CI -0.0051, 0.0135]; crosses zero: yes.
- DeltaAUROC Mixed-type CEM - Black-box = -0.0036 [95% CI -0.0128, 0.0052]; crosses zero: yes.
- DeltaAUROC Standard CBM - Black-box = -0.0127 [95% CI -0.0219, -0.0036]; crosses zero: no.
- DeltaAUROC Learned-softmax GAM - Mixed-type CEM = 0.0078 [95% CI -0.0008, 0.0178]; crosses zero: yes.
- DeltaAUROC Learned-softmax GAM - Standard CBM = 0.0169 [95% CI 0.0064, 0.0273]; crosses zero: no.
- DeltaAUROC Mixed-type CEM - Standard CBM = 0.0091 [95% CI -0.0020, 0.0203]; crosses zero: yes.

## 10. Concept Prediction Results

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Standard CBM / subtlety: MAE 0.1690; RMSE 0.2233; Pearson 0.5673; Spearman 0.5733; N 2633.
- Standard CBM / internalStructure: soft CE 0.0390; Brier 0.0069; macro-F1 0.2497; soft N 2633; hard N 2625; ties 8.
- Standard CBM / calcification: soft CE 0.2072; Brier 0.0491; macro-F1 0.3138; soft N 2633; hard N 2578; ties 55.
- Standard CBM / sphericity: MAE 0.1386; RMSE 0.1767; Pearson 0.4366; Spearman 0.4528; N 2633.
- Standard CBM / margin: MAE 0.1456; RMSE 0.1972; Pearson 0.6868; Spearman 0.6473; N 2633.
- Standard CBM / lobulation: MAE 0.1310; RMSE 0.1857; Pearson 0.4597; Spearman 0.4559; N 2633.
- Standard CBM / spiculation: MAE 0.1211; RMSE 0.1816; Pearson 0.5069; Spearman 0.4164; N 2633.
- Standard CBM / texture: MAE 0.1161; RMSE 0.1802; Pearson 0.7828; Spearman 0.5806; N 2633.
- Mixed-type CEM / subtlety: MAE 0.2038; RMSE 0.2502; Pearson 0.4445; Spearman 0.4491; N 2633.
- Mixed-type CEM / internalStructure: soft CE 0.0827; Brier 0.0136; macro-F1 0.2497; soft N 2633; hard N 2625; ties 8.
- Mixed-type CEM / calcification: soft CE 0.2624; Brier 0.0677; macro-F1 0.3098; soft N 2633; hard N 2578; ties 55.
- Mixed-type CEM / sphericity: MAE 0.1780; RMSE 0.2171; Pearson 0.0561; Spearman 0.0514; N 2633.
- Mixed-type CEM / margin: MAE 0.2463; RMSE 0.2921; Pearson 0.1880; Spearman 0.1709; N 2633.
- Mixed-type CEM / lobulation: MAE 0.1829; RMSE 0.2175; Pearson 0.2701; Spearman 0.2488; N 2633.
- Mixed-type CEM / spiculation: MAE 0.1799; RMSE 0.2155; Pearson 0.3050; Spearman 0.2743; N 2633.
- Mixed-type CEM / texture: MAE 0.2650; RMSE 0.3052; Pearson 0.2025; Spearman 0.1666; N 2633.
- Learned-softmax GAM / subtlety: MAE 0.1677; RMSE 0.2209; Pearson 0.5785; Spearman 0.5823; N 2633.
- Learned-softmax GAM / internalStructure: soft CE 0.0381; Brier 0.0073; macro-F1 0.3121; soft N 2633; hard N 2625; ties 8.
- Learned-softmax GAM / calcification: soft CE 0.2013; Brier 0.0481; macro-F1 0.3127; soft N 2633; hard N 2578; ties 55.
- Learned-softmax GAM / sphericity: MAE 0.1400; RMSE 0.1784; Pearson 0.4242; Spearman 0.4351; N 2633.
- Learned-softmax GAM / margin: MAE 0.1455; RMSE 0.1983; Pearson 0.6834; Spearman 0.6478; N 2633.
- Learned-softmax GAM / lobulation: MAE 0.1272; RMSE 0.1896; Pearson 0.4509; Spearman 0.4721; N 2633.
- Learned-softmax GAM / spiculation: MAE 0.1144; RMSE 0.1838; Pearson 0.4985; Spearman 0.4561; N 2633.
- Learned-softmax GAM / texture: MAE 0.1112; RMSE 0.1791; Pearson 0.7861; Spearman 0.5829; N 2633.

## 11. Contribution Centering

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Standard CBM centered rating contributions: subtlety=0.2852/internalStructure=0.2811/calcification=0.4005/sphericity=-0.0722/margin=0.1375/lobulation=0.2458/spiculation=0.1587/texture=-0.1215; most positive calcification=0.4005; most negative texture=-0.1215.
- Mixed-type CEM centered rating contributions: subtlety=0.1326/internalStructure=0.4861/calcification=0.2635/sphericity=0.1549/margin=0.1234/lobulation=0.2787/spiculation=0.1412/texture=0.1013; most positive internalStructure=0.4861; no negative pooled mean; smallest positive texture=0.1013.
- Learned-softmax GAM centered rating contributions: subtlety=0.2286/internalStructure=0.0413/calcification=0.4022/sphericity=0.1062/margin=0.3702/lobulation=-0.0159/spiculation=0.3229/texture=0.1904; most positive calcification=0.4022; most negative lobulation=-0.0159.

## 12. Concept Intervention

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Standard CBM 100-permutation mean k=0/1/2/3/4/5/6/7/8 MAE: 0.5021/0.5010/0.5001/0.4998/0.4996/0.4998/0.5023/0.5041/0.5075.
- Standard CBM permutation iMAE 0.5014; Delta_iMAE 0.0006 (positive denotes improvement).
- Standard CBM 100-permutation mean k=0/1/2/3/4/5/6/7/8 AUROC: 0.9327/0.9321/0.9311/0.9304/0.9295/0.9287/0.9271/0.9257/0.9237.
- Standard CBM permutation iAUC 0.9291; Delta_iAUC -0.0036 (positive denotes improvement).
- Standard CBM error-first k=0/1/2/3/4/5/6/7/8 MAE: 0.5021/0.4952/0.5082/0.5070/0.5096/0.5080/0.5075/0.5074/0.5075.
- Standard CBM iMAE 0.5060; Delta_iMAE -0.0039 (positive denotes improvement).
- Standard CBM error-first k=0/1/2/3/4/5/6/7/8 AUROC: 0.9327/0.9265/0.9186/0.9166/0.9181/0.9215/0.9236/0.9239/0.9237.
- Standard CBM iAUC 0.9221; Delta_iAUC -0.0106 (positive denotes improvement).
- Mixed-type CEM 100-permutation mean k=0/1/2/3/4/5/6/7/8 MAE: 0.4841/0.4755/0.4681/0.4601/0.4540/0.4483/0.4434/0.4391/0.4358.
- Mixed-type CEM permutation iMAE 0.4560; Delta_iMAE 0.0281 (positive denotes improvement).
- Mixed-type CEM 100-permutation mean k=0/1/2/3/4/5/6/7/8 AUROC: 0.9417/0.9456/0.9493/0.9523/0.9549/0.9573/0.9598/0.9617/0.9638.
- Mixed-type CEM permutation iAUC 0.9542; Delta_iAUC 0.0125 (positive denotes improvement).
- Mixed-type CEM error-first k=0/1/2/3/4/5/6/7/8 MAE: 0.4841/0.4585/0.4484/0.4425/0.4400/0.4367/0.4352/0.4356/0.4358.
- Mixed-type CEM iMAE 0.4446; Delta_iMAE 0.0395 (positive denotes improvement).
- Mixed-type CEM error-first k=0/1/2/3/4/5/6/7/8 AUROC: 0.9417/0.9529/0.9593/0.9600/0.9618/0.9631/0.9637/0.9637/0.9638.
- Mixed-type CEM iAUC 0.9597; Delta_iAUC 0.0180 (positive denotes improvement).
- Learned-softmax GAM 100-permutation mean k=0/1/2/3/4/5/6/7/8 MAE: 0.4804/0.4775/0.4764/0.4760/0.4787/0.4827/0.4891/0.4956/0.5068.
- Learned-softmax GAM permutation iMAE 0.4837; Delta_iMAE -0.0033 (positive denotes improvement).
- Learned-softmax GAM 100-permutation mean k=0/1/2/3/4/5/6/7/8 AUROC: 0.9493/0.9493/0.9491/0.9479/0.9450/0.9423/0.9388/0.9354/0.9298.
- Learned-softmax GAM permutation iAUC 0.9434; Delta_iAUC -0.0059 (positive denotes improvement).
- Learned-softmax GAM error-first k=0/1/2/3/4/5/6/7/8 MAE: 0.4804/0.4645/0.4901/0.4998/0.5049/0.5059/0.5070/0.5068/0.5068.
- Learned-softmax GAM iMAE 0.4966; Delta_iMAE -0.0161 (positive denotes improvement).
- Learned-softmax GAM error-first k=0/1/2/3/4/5/6/7/8 AUROC: 0.9493/0.9464/0.9322/0.9258/0.9245/0.9282/0.9298/0.9297/0.9298.
- Learned-softmax GAM iAUC 0.9320; Delta_iAUC -0.0173 (positive denotes improvement).

## 13. Learned GAM Alpha

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Fold 0 maximum expert weights: subtlety=0.2062/internalStructure=0.2003/calcification=0.2044/sphericity=0.2017/margin=0.2026/lobulation=0.2035/spiculation=0.2065/texture=0.2007.
- Fold 1 maximum expert weights: subtlety=0.2034/internalStructure=0.2012/calcification=0.2107/sphericity=0.2016/margin=0.2016/lobulation=0.2056/spiculation=0.2050/texture=0.2021.
- Fold 2 maximum expert weights: subtlety=0.2018/internalStructure=0.2009/calcification=0.2056/sphericity=0.2016/margin=0.2017/lobulation=0.2056/spiculation=0.2073/texture=0.2014.
- Fold 3 maximum expert weights: subtlety=0.2021/internalStructure=0.2008/calcification=0.2029/sphericity=0.2014/margin=0.2030/lobulation=0.2136/spiculation=0.2034/texture=0.2009.
- Fold 4 maximum expert weights: subtlety=0.2026/internalStructure=0.2006/calcification=0.2064/sphericity=0.2019/margin=0.2013/lobulation=0.2059/spiculation=0.2029/texture=0.2015.

## 14. Grad-CAM Methods

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Grad-CAM used spatial-mean gradients, weighted activations, ReLU, and trilinear upsampling to 64^3 [5].
- Raw FP32 maps were stored without normalization; all-zero post-ReLU maps were explicitly undefined.
- Occlusion preserved output_sensitivity and error_increase separately for saliency and 20 matched-random masks.

## 15. Grad-CAM Accounting

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Requested 73724 = valid 66769 + undefined 6955; undefined rate 0.0943.
- Black-box: valid 2429; undefined 204; rate 0.0775.
- Black-box fold 0: valid 479; undefined 0; rate 0.0000.
- Black-box fold 1: valid 502; undefined 0; rate 0.0000.
- Black-box fold 2: valid 539; undefined 0; rate 0.0000.
- Black-box fold 3: valid 448; undefined 101; rate 0.1840.
- Black-box fold 4: valid 461; undefined 103; rate 0.1826.
- Standard CBM: valid 22413; undefined 1284; rate 0.0542.
- Standard CBM fold 0: valid 3817; undefined 494; rate 0.1146.
- Standard CBM fold 1: valid 4218; undefined 300; rate 0.0664.
- Standard CBM fold 2: valid 4591; undefined 260; rate 0.0536.
- Standard CBM fold 3: valid 4888; undefined 53; rate 0.0107.
- Standard CBM fold 4: valid 4899; undefined 177; rate 0.0349.
- Mixed-type CEM: valid 20316; undefined 3381; rate 0.1427.
- Mixed-type CEM fold 0: valid 3474; undefined 837; rate 0.1942.
- Mixed-type CEM fold 1: valid 3992; undefined 526; rate 0.1164.
- Mixed-type CEM fold 2: valid 4182; undefined 669; rate 0.1379.
- Mixed-type CEM fold 3: valid 4735; undefined 206; rate 0.0417.
- Mixed-type CEM fold 4: valid 3933; undefined 1143; rate 0.2252.
- Learned-softmax GAM: valid 21611; undefined 2086; rate 0.0880.
- Learned-softmax GAM fold 0: valid 4067; undefined 244; rate 0.0566.
- Learned-softmax GAM fold 1: valid 4298; undefined 220; rate 0.0487.
- Learned-softmax GAM fold 2: valid 4515; undefined 336; rate 0.0693.
- Learned-softmax GAM fold 3: valid 4064; undefined 877; rate 0.1775.
- Learned-softmax GAM fold 4: valid 4667; undefined 409; rate 0.0806.
- The complete model x fold x target/concept breakdown is retained in Table 7 (gradcam_accounting.csv).

## 16. Spatial Faithfulness

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Black-box output_sensitivity: saliency mean 0.0252; saliency-random mean -0.3225; 95% range [-0.7498, -0.0148]; saliency > random mean rate 0.0086.
- Black-box error_increase: saliency mean 0.0029; saliency-random mean -0.2354; 95% range [-0.7076, 0.1958]; saliency > random mean rate 0.1441.
- Standard CBM output_sensitivity: saliency mean 0.1494; saliency-random mean -1.0642; 95% range [-3.2466, 0.0454]; saliency > random mean rate 0.0372.
- Standard CBM error_increase: saliency mean -0.0698; saliency-random mean 0.1330; 95% range [-2.5445, 2.3398]; saliency > random mean rate 0.5447.
- Mixed-type CEM output_sensitivity: saliency mean 0.0569; saliency-random mean -0.3735; 95% range [-1.5450, 0.0348]; saliency > random mean rate 0.0579.
- Mixed-type CEM error_increase: saliency mean -0.0285; saliency-random mean -0.0372; 95% range [-1.2431, 0.8668]; saliency > random mean rate 0.4874.
- Learned-softmax GAM output_sensitivity: saliency mean 0.1919; saliency-random mean -1.1230; 95% range [-3.3577, 0.0803]; saliency > random mean rate 0.0447.
- Learned-softmax GAM error_increase: saliency mean -0.1031; saliency-random mean 0.1941; 95% range [-2.3412, 2.6131]; saliency > random mean rate 0.5263.
- Complete fold-target, pooled-target, and pooled-model results are retained in Table 8 (spatial_faithfulness.csv).

## 17. Execution Provenance

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- The execution registry contains 40 immutable model/fold records spanning P5-P9.
- P5-P8 each retained exactly 1 valid committed test evaluation per fold; P9 created 0 additional test evaluations.
- Verifier recoveries are distinguished from scientific execution and do not change persisted predictions or metrics.
- P9 spatial_stage_a: job 8986164; NVIDIA H200; Exit_status 0; scientific status PASS.
- P9 aggregate_invalidated_attempt: job 8987452; CPU-only; Exit_status 1; scientific status INVALIDATED_AGGREGATE_ATTEMPT.
- P9 aggregate_verifier_recovery: job 8987554; CPU-only; Exit_status 0; scientific status PASS.

## 18. Storage and Reproducibility

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- All public values, CIs, tables, and figures derive from 1 shared report_data.json model.
- GitHub excludes checkpoints, private predictions, raw Grad-CAM maps, CT/ROI volumes, UIDs, and patient keys.
- The private archive is verified file-by-file with SHA-256 and is stored only on the Mac.
- Verified private archive: 1698 files; 14386651621 bytes; manifest 67731d14d26d5ff1cbbf36afa903490662f7c130abbc76421a3ebd39edf37df4.
- Table 1: primary_secondary_metrics.csv.
- Table 2: paired_comparisons.csv.
- Table 3: concept_metrics.csv.
- Table 4: intervention_curves.csv.
- Table 5: centered_contributions.csv.
- Table 6: learned_gam_alpha.csv.
- Table 7: gradcam_accounting.csv.
- Table 8: spatial_faithfulness.csv.
- Table 9: execution_registry.csv.

## 19. Negative Findings

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Learned-softmax GAM had the lowest pooled MAE point estimate, but not every paired CI excluded 0.
- Intervention benefit was model-dependent; positive improvement was not uniform across k=0-8.
- Saliency was not uniformly more faithful than matched random masks for either faithfulness quantity.
- Unclipped Black-box scores extended below 1 and above 5; Standard CBM and Mixed-type CEM also produced a small fraction below 1, whereas Learned-softmax GAM stayed within the rating range.

## 20. Limitations

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, limiting claims about clinical cancer detection.
- The study evaluates one patient-grouped five-fold cohort and does not establish external-site generalization.
- The exact pre-ReLU CAM, gradient, activation, and channel-weight decomposition was not persisted, so the 6,955 zero maps cannot be mechanistically resolved without a prohibited new forward pass.
- Occlusion on normalized-zero voxels is a registered perturbation test, not a causal explanation of malignancy.
- This research benchmark is not a clinical diagnostic product.

## 21. Conclusions

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- Learned-softmax GAM achieved the strongest primary point estimate; uncertainty and spatial limitations remain material.
- The findings support research comparison only and do not establish clinical diagnostic utility.

## 22. References

All results were reconstructed read-only from frozen P5–P9 evidence. Primary predictions were not clipped. LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and this system is not a clinical diagnostic product.

- [1] S. G. Armato III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans," Med. Phys., vol. 38, no. 2, pp. 915-931, 2011, doi: 10.1118/1.3528204.
- [2] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4700-4708, 2017, doi: 10.1109/CVPR.2017.243.
- [3] P. W. Koh et al., "Concept Bottleneck Models," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, pp. 5338-5348, 2020.
- [4] M. Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off," in Adv. Neural Inf. Process. Syst., vol. 35, pp. 21400-21413, 2022.
- [5] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 618-626, 2017, doi: 10.1109/ICCV.2017.74.
- [6] B. Efron, "Bootstrap Methods: Another Look at the Jackknife," Ann. Stat., vol. 7, no. 1, pp. 1-26, 1979, doi: 10.1214/aos/1176344552.

## Scientific conclusion codes

- `GAM_LOWEST_POINT_ESTIMATE_MAE`
- `PAIRED_MAE_SUPPORTS_GAM_OVER_BLACKBOX_AND_CBM`
- `AUROC_DIFFERENCES_MOSTLY_UNCERTAIN`
- `INTERVENTION_BENEFIT_MODEL_DEPENDENT`
- `SALIENCY_NOT_UNIFORMLY_MORE_FAITHFUL_THAN_RANDOM`
- `SYSTEMATIC_MODEL_TARGET_ZERO_MAP_LIMITATION`
