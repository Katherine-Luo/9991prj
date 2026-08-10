# LIDC-IDRI Baseline 协议索引

本文件用于消除协议版本歧义。当前开发必须同时以 `docs/PROJECT_STATUS.md` 的 active pointers 与本表为准。

| Version | Status | Requirements | Config | Usage |
|---|---|---|---|---|
| Baseline-v2 | `ACTIVE` | [V2 requirements](./LIDC_IDRI_BASELINE_V2_REQUIREMENTS.md) | [`baseline_v2.yaml`](../configs/baseline_v2.yaml) | 所有后续开发和正式实验 |
| Baseline-v1 | `SUPERSEDED` | [V1 requirements](./LIDC_IDRI_BASELINE_V1_REQUIREMENTS.md) | [`baseline_v1.yaml`](../configs/baseline_v1.yaml) | 仅用于历史复现、Bug 和 provenance 审计 |

## 使用规则

- 仓库只维护一个实时状态文档：`docs/PROJECT_STATUS.md`。
- `SUPERSEDED` 不表示删除或否定历史验收；Baseline-v1 的 P0、P1、P2 完成记录继续有效。
- 不得用 Baseline-v1 指导 P3–P10 的新实现。
- P1 DICOM/XML 审计属于共享数据证据，可由 V2 直接引用。
- V2 使用既有 stable `nodule_uid`，但 cohort 任务语义、配置哈希和后续实验产物必须使用独立版本命名空间。
- 任何未来科学协议变更必须创建新的版本，不得覆盖 V1 或 V2。
