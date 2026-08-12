# Baseline-v2：LIDC-IDRI可解释三维模型的统一评估

**REPORT-DATA-SHA256:** `d9ae6d45308912b9dd6b23c26918bbcbb3790244d6a2575cd5b64d236d8d2807`

## 1. 摘要

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- 2,633个结节；868名患者；五折计数479/502/539/549/564；患者泄漏0。
- Grad-CAM：73,724张请求图 = 66,769张有效图 + 6,955张ReLU后全零未定义图。
- Bootstrap：2,000次患者聚类抽样。
- Black-box: MAE 0.5006 [0.4829, 0.5195]; RMSE 0.6422; AUROC 0.9453; AUPRC 0.8937.
- Standard CBM: MAE 0.5021 [0.4828, 0.5223]; RMSE 0.6496; AUROC 0.9327; AUPRC 0.8657.
- Mixed-type CEM: MAE 0.4841 [0.4669, 0.5021]; RMSE 0.6283; AUROC 0.9417; AUPRC 0.8774.
- Learned-softmax GAM: MAE 0.4804 [0.4623, 0.4985]; RMSE 0.6176; AUROC 0.9493; AUPRC 0.9026.

## 2. 引言

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- LIDC恶性度是放射科医师评估，并非病理确诊 [1]。
- 本系统是研究基准，并非临床诊断产品。
- 各模型使用DenseNet-121 [2]；CBM与CEM术语沿用 [3]、[4]，Mixed-type CEM与Learned-softmax GAM则指本项目预注册的特定设计。

## 3. 方法

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- 主要分数未经截断；2,000次共享患者聚类Bootstrap抽样生成百分位95%置信区间 [6]。
- 次要任务Youden-J阈值仅使用各折验证集中恶性度 <= 2或 >= 4的样本。
- 每张有效Grad-CAM图 [5] 使用26,215个显著体素和20个匹配随机遮罩。
- 配对符号定义为DeltaMAE=MAE_A-MAE_B、DeltaAUROC=AUROC_B-AUROC_A；正值表示模型B更优。
- 干预符号定义为Delta_iMAE=baseline_MAE-iMAE、Delta_iAUC=iAUC-baseline_AUROC；正值表示改善。

## 4. 队列与完整性

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- 标准OOF队列包含2,633个唯一结节，来自868名患者。
- 外层测试折计数为479/502/539/549/564；患者泄漏为0。
- 全部4个模型使用相同目标值与逐折一致的外层测试成员。

## 5. 主要与次要结果

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- Black-box：MAE 0.5006 [95% CI 0.4829, 0.5195]; RMSE 0.6422 [95% CI 0.6189, 0.6671]; normalized MAE 0.1252 [95% CI 0.1207, 0.1299]; Pearson 0.7157 [95% CI 0.6894, 0.7409]; Spearman 0.6345 [95% CI 0.5994, 0.6676]; unclipped 1-5 prediction range [0.4885, 5.1200]; below 1 rate 0.0148; above 5 rate 0.0011.
- Standard CBM：MAE 0.5021 [95% CI 0.4828, 0.5223]; RMSE 0.6496 [95% CI 0.6254, 0.6749]; normalized MAE 0.1255 [95% CI 0.1207, 0.1306]; Pearson 0.7076 [95% CI 0.6770, 0.7354]; Spearman 0.6091 [95% CI 0.5700, 0.6482]; unclipped 1-5 prediction range [0.8585, 4.5798]; below 1 rate 0.0087; above 5 rate 0.0000.
- Mixed-type CEM：MAE 0.4841 [95% CI 0.4669, 0.5021]; RMSE 0.6283 [95% CI 0.6040, 0.6537]; normalized MAE 0.1210 [95% CI 0.1167, 0.1255]; Pearson 0.7296 [95% CI 0.7010, 0.7565]; Spearman 0.6400 [95% CI 0.6036, 0.6734]; unclipped 1-5 prediction range [0.8229, 4.9351]; below 1 rate 0.0042; above 5 rate 0.0000.
- Learned-softmax GAM：MAE 0.4804 [95% CI 0.4623, 0.4985]; RMSE 0.6176 [95% CI 0.5924, 0.6423]; normalized MAE 0.1201 [95% CI 0.1156, 0.1246]; Pearson 0.7405 [95% CI 0.7117, 0.7676]; Spearman 0.6528 [95% CI 0.6160, 0.6883]; unclipped 1-5 prediction range [1.0081, 4.6816]; below 1 rate 0.0000; above 5 rate 0.0000.
- DeltaMAE Black-box - Learned-softmax GAM = 0.0201 [95% CI 0.0101, 0.0305]；跨零：否。
- DeltaMAE Black-box - Mixed-type CEM = 0.0164 [95% CI 0.0060, 0.0272]；跨零：否。
- DeltaMAE Black-box - Standard CBM = -0.0016 [95% CI -0.0148, 0.0120]；跨零：是。
- DeltaMAE Mixed-type CEM - Learned-softmax GAM = 0.0037 [95% CI -0.0059, 0.0130]；跨零：是。
- DeltaMAE Standard CBM - Learned-softmax GAM = 0.0217 [95% CI 0.0109, 0.0326]；跨零：否。
- DeltaMAE Standard CBM - Mixed-type CEM = 0.0181 [95% CI 0.0059, 0.0301]；跨零：否。
- Black-box：AUROC 0.9453 [95% CI 0.9264, 0.9615]; AUPRC 0.8937 [95% CI 0.8586, 0.9250].
- Standard CBM：AUROC 0.9327 [95% CI 0.9113, 0.9510]; AUPRC 0.8657 [95% CI 0.8263, 0.8995].
- Mixed-type CEM：AUROC 0.9417 [95% CI 0.9202, 0.9605]; AUPRC 0.8774 [95% CI 0.8332, 0.9155].
- Learned-softmax GAM：AUROC 0.9493 [95% CI 0.9266, 0.9685]; AUPRC 0.9026 [95% CI 0.8678, 0.9340].
- DeltaAUROC Learned-softmax GAM - Black-box = 0.0042 [95% CI -0.0051, 0.0135]；跨零：是。
- DeltaAUROC Mixed-type CEM - Black-box = -0.0036 [95% CI -0.0128, 0.0052]；跨零：是。
- DeltaAUROC Standard CBM - Black-box = -0.0127 [95% CI -0.0219, -0.0036]；跨零：否。
- DeltaAUROC Learned-softmax GAM - Mixed-type CEM = 0.0078 [95% CI -0.0008, 0.0178]；跨零：是。
- DeltaAUROC Learned-softmax GAM - Standard CBM = 0.0169 [95% CI 0.0064, 0.0273]；跨零：否。
- DeltaAUROC Mixed-type CEM - Standard CBM = 0.0091 [95% CI -0.0020, 0.0203]；跨零：是。

## 6. 概念预测结果

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- Standard CBM / subtlety：MAE 0.1690; RMSE 0.2233; Pearson 0.5673; Spearman 0.5733; N 2633.
- Standard CBM / internalStructure：soft CE 0.0390; Brier 0.0069; macro-F1 0.2497; soft N 2633; hard N 2625; ties 8.
- Standard CBM / calcification：soft CE 0.2072; Brier 0.0491; macro-F1 0.3138; soft N 2633; hard N 2578; ties 55.
- Standard CBM / sphericity：MAE 0.1386; RMSE 0.1767; Pearson 0.4366; Spearman 0.4528; N 2633.
- Standard CBM / margin：MAE 0.1456; RMSE 0.1972; Pearson 0.6868; Spearman 0.6473; N 2633.
- Standard CBM / lobulation：MAE 0.1310; RMSE 0.1857; Pearson 0.4597; Spearman 0.4559; N 2633.
- Standard CBM / spiculation：MAE 0.1211; RMSE 0.1816; Pearson 0.5069; Spearman 0.4164; N 2633.
- Standard CBM / texture：MAE 0.1161; RMSE 0.1802; Pearson 0.7828; Spearman 0.5806; N 2633.
- Mixed-type CEM / subtlety：MAE 0.2038; RMSE 0.2502; Pearson 0.4445; Spearman 0.4491; N 2633.
- Mixed-type CEM / internalStructure：soft CE 0.0827; Brier 0.0136; macro-F1 0.2497; soft N 2633; hard N 2625; ties 8.
- Mixed-type CEM / calcification：soft CE 0.2624; Brier 0.0677; macro-F1 0.3098; soft N 2633; hard N 2578; ties 55.
- Mixed-type CEM / sphericity：MAE 0.1780; RMSE 0.2171; Pearson 0.0561; Spearman 0.0514; N 2633.
- Mixed-type CEM / margin：MAE 0.2463; RMSE 0.2921; Pearson 0.1880; Spearman 0.1709; N 2633.
- Mixed-type CEM / lobulation：MAE 0.1829; RMSE 0.2175; Pearson 0.2701; Spearman 0.2488; N 2633.
- Mixed-type CEM / spiculation：MAE 0.1799; RMSE 0.2155; Pearson 0.3050; Spearman 0.2743; N 2633.
- Mixed-type CEM / texture：MAE 0.2650; RMSE 0.3052; Pearson 0.2025; Spearman 0.1666; N 2633.
- Learned-softmax GAM / subtlety：MAE 0.1677; RMSE 0.2209; Pearson 0.5785; Spearman 0.5823; N 2633.
- Learned-softmax GAM / internalStructure：soft CE 0.0381; Brier 0.0073; macro-F1 0.3121; soft N 2633; hard N 2625; ties 8.
- Learned-softmax GAM / calcification：soft CE 0.2013; Brier 0.0481; macro-F1 0.3127; soft N 2633; hard N 2578; ties 55.
- Learned-softmax GAM / sphericity：MAE 0.1400; RMSE 0.1784; Pearson 0.4242; Spearman 0.4351; N 2633.
- Learned-softmax GAM / margin：MAE 0.1455; RMSE 0.1983; Pearson 0.6834; Spearman 0.6478; N 2633.
- Learned-softmax GAM / lobulation：MAE 0.1272; RMSE 0.1896; Pearson 0.4509; Spearman 0.4721; N 2633.
- Learned-softmax GAM / spiculation：MAE 0.1144; RMSE 0.1838; Pearson 0.4985; Spearman 0.4561; N 2633.
- Learned-softmax GAM / texture：MAE 0.1112; RMSE 0.1791; Pearson 0.7861; Spearman 0.5829; N 2633.

## 7. 概念干预

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- Standard CBM 100次排列均值k=0/1/2/3/4/5/6/7/8 MAE：0.5021/0.5010/0.5001/0.4998/0.4996/0.4998/0.5023/0.5041/0.5075。
- Standard CBM排列iMAE 0.5014；Delta_iMAE 0.0006（正值表示改善）。
- Standard CBM 100次排列均值k=0/1/2/3/4/5/6/7/8 AUROC：0.9327/0.9321/0.9311/0.9304/0.9295/0.9287/0.9271/0.9257/0.9237。
- Standard CBM排列iAUC 0.9291；Delta_iAUC -0.0036（正值表示改善）。
- Standard CBM误差优先k=0/1/2/3/4/5/6/7/8 MAE：0.5021/0.4952/0.5082/0.5070/0.5096/0.5080/0.5075/0.5074/0.5075。
- Standard CBM iMAE 0.5060；Delta_iMAE -0.0039（正值表示改善）。
- Standard CBM误差优先k=0/1/2/3/4/5/6/7/8 AUROC：0.9327/0.9265/0.9186/0.9166/0.9181/0.9215/0.9236/0.9239/0.9237。
- Standard CBM iAUC 0.9221；Delta_iAUC -0.0106（正值表示改善）。
- Mixed-type CEM 100次排列均值k=0/1/2/3/4/5/6/7/8 MAE：0.4841/0.4755/0.4681/0.4601/0.4540/0.4483/0.4434/0.4391/0.4358。
- Mixed-type CEM排列iMAE 0.4560；Delta_iMAE 0.0281（正值表示改善）。
- Mixed-type CEM 100次排列均值k=0/1/2/3/4/5/6/7/8 AUROC：0.9417/0.9456/0.9493/0.9523/0.9549/0.9573/0.9598/0.9617/0.9638。
- Mixed-type CEM排列iAUC 0.9542；Delta_iAUC 0.0125（正值表示改善）。
- Mixed-type CEM误差优先k=0/1/2/3/4/5/6/7/8 MAE：0.4841/0.4585/0.4484/0.4425/0.4400/0.4367/0.4352/0.4356/0.4358。
- Mixed-type CEM iMAE 0.4446；Delta_iMAE 0.0395（正值表示改善）。
- Mixed-type CEM误差优先k=0/1/2/3/4/5/6/7/8 AUROC：0.9417/0.9529/0.9593/0.9600/0.9618/0.9631/0.9637/0.9637/0.9638。
- Mixed-type CEM iAUC 0.9597；Delta_iAUC 0.0180（正值表示改善）。
- Learned-softmax GAM 100次排列均值k=0/1/2/3/4/5/6/7/8 MAE：0.4804/0.4775/0.4764/0.4760/0.4787/0.4827/0.4891/0.4956/0.5068。
- Learned-softmax GAM排列iMAE 0.4837；Delta_iMAE -0.0033（正值表示改善）。
- Learned-softmax GAM 100次排列均值k=0/1/2/3/4/5/6/7/8 AUROC：0.9493/0.9493/0.9491/0.9479/0.9450/0.9423/0.9388/0.9354/0.9298。
- Learned-softmax GAM排列iAUC 0.9434；Delta_iAUC -0.0059（正值表示改善）。
- Learned-softmax GAM误差优先k=0/1/2/3/4/5/6/7/8 MAE：0.4804/0.4645/0.4901/0.4998/0.5049/0.5059/0.5070/0.5068/0.5068。
- Learned-softmax GAM iMAE 0.4966；Delta_iMAE -0.0161（正值表示改善）。
- Learned-softmax GAM误差优先k=0/1/2/3/4/5/6/7/8 AUROC：0.9493/0.9464/0.9322/0.9258/0.9245/0.9282/0.9298/0.9297/0.9298。
- Learned-softmax GAM iAUC 0.9320；Delta_iAUC -0.0173（正值表示改善）。

## 8. Grad-CAM计数

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- 请求图73724 = 有效图66769 + 未定义图6955；未定义率0.0943。
- Black-box：有效2429；未定义204；比例0.0775。
- Black-box fold 0：有效479；未定义0；比例0.0000。
- Black-box fold 1：有效502；未定义0；比例0.0000。
- Black-box fold 2：有效539；未定义0；比例0.0000。
- Black-box fold 3：有效448；未定义101；比例0.1840。
- Black-box fold 4：有效461；未定义103；比例0.1826。
- Standard CBM：有效22413；未定义1284；比例0.0542。
- Standard CBM fold 0：有效3817；未定义494；比例0.1146。
- Standard CBM fold 1：有效4218；未定义300；比例0.0664。
- Standard CBM fold 2：有效4591；未定义260；比例0.0536。
- Standard CBM fold 3：有效4888；未定义53；比例0.0107。
- Standard CBM fold 4：有效4899；未定义177；比例0.0349。
- Mixed-type CEM：有效20316；未定义3381；比例0.1427。
- Mixed-type CEM fold 0：有效3474；未定义837；比例0.1942。
- Mixed-type CEM fold 1：有效3992；未定义526；比例0.1164。
- Mixed-type CEM fold 2：有效4182；未定义669；比例0.1379。
- Mixed-type CEM fold 3：有效4735；未定义206；比例0.0417。
- Mixed-type CEM fold 4：有效3933；未定义1143；比例0.2252。
- Learned-softmax GAM：有效21611；未定义2086；比例0.0880。
- Learned-softmax GAM fold 0：有效4067；未定义244；比例0.0566。
- Learned-softmax GAM fold 1：有效4298；未定义220；比例0.0487。
- Learned-softmax GAM fold 2：有效4515；未定义336；比例0.0693。
- Learned-softmax GAM fold 3：有效4064；未定义877；比例0.1775。
- Learned-softmax GAM fold 4：有效4667；未定义409；比例0.0806。
- 完整model x fold x target/concept明细保存在表7（gradcam_accounting.csv）。

## 9. 空间忠实度

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- Black-box output_sensitivity：saliency mean 0.0252; saliency-random mean -0.3225; 95% range [-0.7498, -0.0148]; saliency > random mean rate 0.0086.
- Black-box error_increase：saliency mean 0.0029; saliency-random mean -0.2354; 95% range [-0.7076, 0.1958]; saliency > random mean rate 0.1441.
- Standard CBM output_sensitivity：saliency mean 0.1494; saliency-random mean -1.0642; 95% range [-3.2466, 0.0454]; saliency > random mean rate 0.0372.
- Standard CBM error_increase：saliency mean -0.0698; saliency-random mean 0.1330; 95% range [-2.5445, 2.3398]; saliency > random mean rate 0.5447.
- Mixed-type CEM output_sensitivity：saliency mean 0.0569; saliency-random mean -0.3735; 95% range [-1.5450, 0.0348]; saliency > random mean rate 0.0579.
- Mixed-type CEM error_increase：saliency mean -0.0285; saliency-random mean -0.0372; 95% range [-1.2431, 0.8668]; saliency > random mean rate 0.4874.
- Learned-softmax GAM output_sensitivity：saliency mean 0.1919; saliency-random mean -1.1230; 95% range [-3.3577, 0.0803]; saliency > random mean rate 0.0447.
- Learned-softmax GAM error_increase：saliency mean -0.1031; saliency-random mean 0.1941; 95% range [-2.3412, 2.6131]; saliency > random mean rate 0.5263.
- 完整折-目标、汇总目标及汇总模型结果保存在表8（spatial_faithfulness.csv）。

## 10. 讨论与局限

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- Learned-softmax GAM的pooled MAE点估计最低，但并非所有配对置信区间都排除0。
- 干预收益依赖模型；在k=0-8范围内并非始终呈正向改善。
- 对于两种faithfulness量，显著区域并非始终优于匹配随机遮罩。
- 未经截断的Black-box分数同时低于1并高于5；Standard CBM与Mixed-type CEM也有少量分数低于1，而Learned-softmax GAM保持在评分范围内。
- 全部6,955张未定义图均确认是ReLU后全零，但未保存其完整ReLU前/梯度分解。
- 全零图的集中分布被归类为SYSTEMATIC_MODEL/TARGET_ISSUE，而非已证实的实现缺陷。

## 11. 可复现性

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

- 全部公开数值、置信区间、表格与图均来自1个共享report_data.json数据模型。
- GitHub排除检查点、私有预测、原始Grad-CAM图、CT/ROI体数据、UID及patient key。
- 私有备份按文件逐一使用SHA-256验证，且仅保存在Mac。
- 已验证私有备份：1698个文件；14386651621字节；manifest 67731d14d26d5ff1cbbf36afa903490662f7c130abbc76421a3ebd39edf37df4。
- 表1：primary_secondary_metrics.csv。
- 表2：paired_comparisons.csv。
- 表3：concept_metrics.csv。
- 表4：intervention_curves.csv。
- 表5：centered_contributions.csv。
- 表6：learned_gam_alpha.csv。
- 表7：gradcam_accounting.csv。
- 表8：spatial_faithfulness.csv。
- 表9：execution_registry.csv。

## 12. 参考文献

所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。

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
