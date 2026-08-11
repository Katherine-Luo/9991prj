# LIDC-IDRI Baseline-v2

This repository implements the active [Baseline-v2 protocol](docs/LIDC_IDRI_BASELINE_V2_REQUIREMENTS.md) for reproducible research on LIDC-IDRI pulmonary nodule malignancy scoring. It is a research baseline and is not a clinical diagnostic system.

Protocol status and the current development phase are recorded in [PROJECT_STATUS.md](docs/PROJECT_STATUS.md). Baseline-v1 is retained only for historical provenance; Baseline-v2 is the sole active protocol.

## Model-method declarations

- The P7 CEM is **a project-specific mixed-type extension of the original CEM**. The original method supplies sample-conditioned active/inactive embeddings, shared concept scoring, and concept intervention. This project extends that design to six continuous and two multiclass radiological concept groups under the frozen Baseline-v2 protocol.
- P7 does not claim that its mixed-type state generators, probability targets, joint loss weighting, initialization domains, or exact intervention schedule were reported by the original CEM paper.

See the frozen P7 execution supplement for the complete preregistered implementation choices.
