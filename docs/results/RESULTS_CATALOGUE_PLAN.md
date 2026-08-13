# P10 Results & Artifacts Master Catalogue Plan

## 1. Approval gate

```text
RESULTS_CATALOGUE_PLAN_APPROVED=0
IMPLEMENTATION_AUTHORIZED=0
```

This document is a plan only. Creating or approving it does not build the catalogue, scan private results, regenerate reports, or authorize any P5-P9 scientific computation.

Approval rules:

- User approval must explicitly name `RESULTS_CATALOGUE_PLAN.md`.
- Approval is bound to the exact file SHA-256 reported to the user.
- Any later content change invalidates only this plan's approval and resets `RESULTS_CATALOGUE_PLAN_APPROVED=0`.
- Catalogue implementation may begin only when this plan and `P10_CATALOGUE_DRIVEN_BILINGUAL_REPORT_PLAN.md` are both explicitly approved.
- After catalogue implementation and validation, work must stop for a separate user review of the generated catalogue before report generation begins.

## 2. Objective and immutable boundaries

Build a table-driven ledger answering:

```text
What was run
-> what result was produced
-> where the frozen source is stored
-> what table can be made
-> what figure can be made
-> where it belongs in the final report
```

The catalogue covers P0-P10 for phase continuity, with deliberately asymmetric depth:

```text
P0-P4 = lightweight provenance and index coverage only
P5-P9 = full scientific result and artifact coverage
P10 = reporting, catalogue, archive, and rendered-asset coverage
```

P0-P4 continuity rows must not expand into unnecessary scientific-catalogue complexity or displace the P5-P9 scientific results that the user needs to inspect.

Hard boundaries:

- Do not train a model, create an optimizer, or modify a checkpoint.
- Do not run a model forward pass or test inference.
- Do not reselect checkpoints or rerun any committed test transaction.
- Do not recompute bootstrap draws, intervention results, contributions, Grad-CAM, occlusion, faithfulness, or any other scientific estimate for presentation purposes.
- Do not recompute scientific results when the registered value is absent.
- Read existing frozen values only; deterministic formatting, indexing, and explicitly registered non-destructive display transformations are allowed.
- Do not modify P4 splits/initializations or any P5-P9 checkpoint, history, prediction, metric, evaluation, OOF, intervention, Grad-CAM, occlusion, or faithfulness artifact.
- Do not invent scientific interpretation or convert an unavailable value into a numeric placeholder.
- Do not create or start P11.

## 3. Deliverables

### 3.1 Public deidentified catalogue

```text
docs/results/
  RESULTS_MASTER_CATALOGUE.md
  RESULTS_ARTIFACTS_MASTER_TABLE.md
  results_catalogue_registry.json
  catalogue_manifest.json
  results_master_catalogue.csv
  RESULTS_ARTIFACTS_MASTER_TABLE.csv
  tables_inventory.csv
  figures_inventory.csv
  artifacts_inventory.csv
  catalogue_to_report_plan.csv
  catalogue_tables/
    CAT_A_phase_overview.csv
    CAT_B_training_results.csv
    CAT_C_primary_results.csv
    CAT_D_paired_primary.csv
    CAT_E_secondary_results.csv
    CAT_F_paired_secondary.csv
    CAT_G_continuous_concepts.csv
    CAT_H_categorical_concepts.csv
    CAT_I_interventions.csv
    CAT_J_contributions.csv
    CAT_K_gam_alpha.csv
    CAT_L_gradcam.csv
    CAT_M_undefined_rca.csv
    CAT_N_spatial_faithfulness.csv
    CAT_O_tables.csv
    CAT_P_figures.csv
    CAT_Q_qualitative_cases.csv
    CAT_R_storage.csv
    CAT_S_report_evidence.csv
    CAT_T_gaps.csv
```

`RESULTS_MASTER_CATALOGUE.md` contains section headings, one-line legends, and tables only. It must not become a narrative report.

`RESULTS_ARTIFACTS_MASTER_TABLE.md`, its public CSV twin, and its private XLSX twin are the human-facing top-level views. They must answer, without requiring the user to understand the registry implementation:

```text
What did I run?
What result or artifact did it produce?
Where is the frozen source?
Is it public or private?
Can it render a table?
Can it render a figure?
Has the visualization already been rendered?
Where should it appear in the report?
```

The three views are deterministic projections of `results_catalogue_registry.json`; they are never maintained by hand.

Required human-readable columns:

- `Catalogue Item ID`
- `Phase`
- `Model`
- `Fold`
- `Result / Artifact type`
- `Scientific content`
- `Exists?`
- `Frozen source`
- `Public / Private`
- `Report placement` (`Main report`, `Appendix`, `Private appendix`, `Audit-only`, or `No`)
- `Table renderable?`
- `Figure renderable?`
- `Existing visualization?`
- `Visualization status`
- `New inference required?`
- `Assigned report section`
- `Integrity status`
- `Notes`

### 3.2 Private local catalogue overlay

```text
mac-archive://p10_private_report/
  RESULTS_MASTER_CATALOGUE.xlsx
  RESULTS_ARTIFACTS_MASTER_TABLE.xlsx
  results_catalogue_private_locations.csv
```

The public catalogue uses only root aliases and relative paths:

```text
repo://
katana-run://
mac-archive://
private-report://
```

Only the private overlay may contain exact Mac/Katana absolute paths, restricted identifiers, or sensitive locator mappings. The private overlay is never a Git candidate.

## 4. Canonical data model

`results_catalogue_registry.json` is the canonical normalized registry. Markdown, CSV, and private XLSX are deterministic views of this registry and must never be edited independently.

The authoritative architecture is fixed:

```text
frozen P5-P9 artifacts
-> verified structured report data
-> Results Catalogue registry
-> deterministic Markdown/CSV/private XLSX views
-> report mappings
```

### 4.1 Stable identifiers

Identifiers are semantic and independent of row order:

```text
RES-P9-PRIMARY-BLACKBOX
RES-P8-GAM-FOLD3-TASK
ART-P7-CEM-FOLD2-BEST-CKPT
CAT-TBL-GRADCAM-ACCOUNTING
RPT-T07
RPT-F12
SEC-RESULTS-WHERE
CASE-0001
```

### 4.2 Required registry fields

Every item records:

- `catalogue_item_id`
- `entity_type`
- `phase`
- `model`
- `fold`
- `concept_or_target`
- `result_name`
- `scientific_question`
- `scientific_status`
- `availability_status`
- `report_usage_status`
- `source_artifact_id`
- `source_root_alias`
- `source_relative_path`
- `source_field_path`
- `source_sha256`
- `row_or_sample_count`
- `privacy_class`
- `new_inference_required`
- `report_section_id`
- `report_table_ids`
- `report_figure_ids`
- `omission_reason`
- `approval_reference`
- `integrity_status`

Qualitative and spatial items additionally record, where applicable:

- `case_role`
- `full_ct_slice_available`
- `full_ct_slice_source`
- `full_ct_slice_z_index`
- `roi_bbox_available`
- `roi_bbox_coordinates`
- `zoomed_roi_available`
- `roi_source_available`
- `gradcam_target`
- `gradcam_valid_or_undefined`
- `gradcam_overlay_renderable`
- `full_slice_reprojection_renderable`
- `concept_specific_targets_available`
- `display_windowing_available`
- `display_windowing_policy`
- `visualization_normalization_policy`
- `caption_warning_required`
- `original_frozen_ct_source_available`
- `series_and_slice_provenance_available`
- `roi_to_full_volume_mapping_available`
- `read_only_full_ct_renderable`
- `categorical_reader_vote_distribution_available`
- `categorical_modal_label_displayable`
- `case_level_intervention_evidence_available`
- `new_inference_required`

### 4.3 Controlled states

```text
availability_status:
  RESULT_ALREADY_EXISTS
  VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS
  DATA_NOT_PERSISTED
  WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE

report_usage_status:
  USED_MAIN_TEXT
  USED_APPENDIX
  USED_PRIVATE_APPENDIX
  AUDIT_ONLY
  INTENTIONALLY_OMITTED_WITH_REASON

integrity_status:
  VERIFIED
  MISSING
  HASH_MISMATCH
  NOT_APPLICABLE

qualitative_renderability:
  FULL_CT_CONTEXT_AVAILABLE
  ROI_ONLY_AVAILABLE
  FULL_SLICE_REPROJECTION_AVAILABLE
  FULL_SLICE_REPROJECTION_NOT_AVAILABLE_FROM_FROZEN_DATA
  GRADCAM_AVAILABLE
  GRADCAM_UNDEFINED_ZERO_MAP
  CASE_LEVEL_INTERVENTION_AVAILABLE
  NOT_RENDERABLE_FROM_FROZEN_DATA
```

No scientifically important item may remain unclassified. `INTENTIONALLY_OMITTED_WITH_REASON` requires an explicit user approval reference. `DATA_NOT_PERSISTED` and `WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE` remain unavailable and must not trigger new compute.

## 5. Catalogue tables A-T

### CAT-A - Experimental phase overview

Include 11 phase-summary rows for P0-P10 and child rows for the P9 task evaluation, bootstrap, concept fidelity, intervention, contribution centering, Grad-CAM, occlusion faithfulness, and undefined-map RCA components.

P0-P4 rows remain lightweight provenance/index entries. P5-P9 rows carry full scientific/artifact detail, and P10 rows cover catalogue, report, archive, figures, tables, QA, and approval evidence.

Columns:

```text
Phase | Component | Purpose | Model/analysis | Input | Main output |
Scientific status | Final artifact location | Used in final report
```

### CAT-B - Four-model training result inventory

Include 20 P5-P8 model x fold rows and one pooled OOF row per model.

```text
Phase | Model | Fold | N test | Best epoch | Best validation objective |
Test MAE | Test RMSE | Pearson | Spearman | Checkpoint | Predictions |
Metrics | Evaluation | Test transaction count | Status
```

Scheduler terminal status and scientific artifact status remain distinct so verifier false failures or PBS holds are not rewritten as training failures.

### CAT-C - Primary task results

Four model rows with 2,633 nodules, 868 patients, all registered primary point estimates, all existing 2,000 patient-bootstrap intervals, unclipped prediction range/rates, and MAE rank.

### CAT-D - Paired primary comparisons

Six model-pair rows. Preserve the registered sign convention, observed delta, percentile interval, zero-crossing flag, and supported direction. Do not add p-values.

### CAT-E - Secondary extreme-task results

Four model rows with 1,073 extreme nodules / 578 patients, AUROC/AUPRC and intervals, fold-validation-extreme-only threshold rule, and available threshold-dependent metrics. Unavailable fields use controlled availability states rather than invented numbers.

### CAT-F - Paired secondary comparisons

Six model-pair rows with registered `Delta_AUROC`, interval, zero-crossing flag, and supported direction.

### CAT-G - Continuous concept results

Exactly 18 model x continuous-concept rows for Standard CBM, Mixed-type CEM, and Learned-softmax GAM across subtlety, sphericity, margin, lobulation, spiculation, and texture.

### CAT-H - Categorical concept results

Exactly 6 model x categorical-concept rows for internalStructure and calcification, preserving soft N, hard N, ties, soft cross-entropy, multiclass Brier, and tie-excluded macro-F1.

### CAT-I - Concept intervention inventory

Six rows covering each concept model under random/permutation and error-first ordering. Store the complete `k=0..8` MAE/AUROC curves, iMAE/Delta_iMAE, iAUC/Delta_iAUC, and overall direction. Negative intervention results remain visible.

### CAT-J - Contribution inventory

Exactly 24 model x concept rows. Distinguish:

- train-fold centering constant;
- centered OOF contribution;
- mean absolute contribution.

Record whether an empirical contribution profile and case-level contribution bar can be constructed from frozen artifacts. Centering constants must never be labelled importance.

### CAT-K - GAM learned-alpha inventory

Exactly 40 fold x concept rows. Record five initial and five final weights, min/max, simplex verification, and the source of the complete vector rather than only its maximum.

### CAT-L - Grad-CAM inventory

Include:

- 140 model x fold x target detail rows;
- 28 pooled model x target rows;
- one global accounting row.

The global identity must be:

```text
73,724 requested = 66,769 valid + 6,955 undefined
```

Record raw FP32 storage, occlusion availability, private shard alias, and qualitative CT/overlay feasibility.

Do not reduce feasibility to a single yes/no field. For each relevant model/fold/target/case binding, register full-slice availability and source, frozen z index, ROI source and bounding box, zoomed ROI availability, map valid/undefined state, display-window policy, ROI-overlay renderability, full-slice reprojection renderability, concept-specific target availability, display-only normalization policy, required caption warning, and whether a missing component would require prohibited new inference.

A full axial CT visualization may be rendered read-only from the original frozen CT/DICOM series referenced by existing provenance only when all three conditions are verified:

1. the original frozen CT source still exists and is readable without changing it;
2. exact series and slice provenance for the case exists;
3. the frozen ROI-to-full-volume coordinate mapping can be recovered exactly.

This is a display operation, not model inference. If any condition fails, full-slice context or reprojection is marked `FULL_SLICE_REPROJECTION_NOT_AVAILABLE_FROM_FROZEN_DATA`, `DATA_NOT_PERSISTED`, or `NOT_RENDERABLE_FROM_FROZEN_DATA` as appropriate. No series, slice, ROI location, or heatmap placement may be inferred heuristically.

### CAT-M - Undefined-map RCA inventory

Include 28 model x target rows with counts/rates, fold/class concentration, confirmed post-ReLU zero status, implementation-bug evidence, exact-mechanism availability, and RCA classification. Pre-ReLU/gradient quantities not persisted are `NOT_PERSISTED / UNRESOLVED`.

### CAT-N - Spatial faithfulness inventory

Include model x target, model-pooled, and global rows. Keep output sensitivity and error increase separate, each with saliency, matched-random, saliency-minus-random, and saliency-greater-than-random rate. Interpretation uses short controlled labels only.

### CAT-O - Scientific table inventory

Inventory every existing table and every planned report table, including current P10 machine-readable CSVs, report tables `RPT-T01` through `RPT-T18`, the frozen-case index `RPT-TA01`, and the case-level concept/malignancy prediction table `RPT-TA02`.

For `RPT-TA02`, register the frozen source for malignancy prediction/target and every concept prediction/reader target. Categorical targets must preserve the complete frozen reader vote distribution; a modal reader label may be registered as a display-only convenience but must never replace or misrepresent the distributional target.

### CAT-P - Scientific figure inventory

Inventory every existing figure and every planned figure that can be rendered from frozen artifacts. Record question, source, existing path, revision need, privacy, report section, and whether new inference is required.

For every qualitative figure component, record whether frozen evidence supports:

1. original full axial CT slice;
2. full CT slice with ROI bounding box;
3. zoomed ROI crop;
4. ROI plus Grad-CAM overlay;
5. full-slice Grad-CAM reprojection;
6. malignancy Grad-CAM;
7. spiculation Grad-CAM;
8. margin Grad-CAM;
9. texture Grad-CAM;
10. case-level concept prediction-versus-GT table;
11. case-level centered contribution bars;
12. undefined/zero-map limitation panel;
13. persisted case-level intervention before/after evidence;
14. an integrated Prediction-WHERE-WHAT-WHY-HOW case panel.

Each component receives a controlled renderability state. Missing components must be `DATA_NOT_PERSISTED` or `WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE`; they must never be silently synthesized.

### CAT-Q - Qualitative case inventory

Include the 14 frozen cases using opaque `CASE-####` labels only. Record role, model, fold, available prediction/concept/contribution/Grad-CAM evidence, map status, and intended panel. UID/patient mappings remain private.

For every frozen case and target, also record `case_role` (`representative`, `failure`, `intervention_worsening`, `limitation`, `undefined_zero_map`, or `integrated_explanation`), full CT/ROI/z-index/bounding-box metadata, zoomed-ROI availability, display windowing, Grad-CAM validity, ROI-overlay and full-slice-reprojection feasibility, concept-specific map availability, caption warnings, and the complete fourteen-component support checklist defined in CAT-P.

For case-level concept evidence, record frozen predictions and targets for all six continuous concepts plus internalStructure and calcification. For each categorical concept, retain the full reader vote distribution and optionally expose a modal label only as a clearly labelled readability aid.

For the integrated case explanation, independently classify the availability of Prediction, WHERE, WHAT, WHY, and HOW. HOW is available only when sufficient case-level intervention evidence was persisted by P9. Otherwise it is `DATA_NOT_PERSISTED`; the Catalogue must not request or imply a recomputed intervention.

The Catalogue must therefore function as a paper-style qualitative-asset manifest, not merely state that a Grad-CAM artifact exists.

### CAT-R - Complete artifact storage map

Cover P5 Black-box, P6 Standard CBM, P7 Mixed-type CEM, P8 Learned-softmax GAM, P9 OOF/intervention/contribution/bootstrap/spatial artifacts, P5-P10 audits, P10 report data/figures/private appendices, and the complete Mac archive.

### CAT-S - Final-report evidence map

Map Dataset, Model design, Prediction, WHERE, WHAT, WHY, HOW, synthesis, Discussion, Limitations, and Reproducibility to required result/table/figure/source IDs and privacy classes.

### CAT-T - Missing or incomplete outputs

Classify expected items as:

```text
RESULT_ALREADY_EXISTS
VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS
DATA_NOT_PERSISTED
WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE
```

Record report-blocking status and the permitted action. No missing item may silently trigger new scientific compute.

## 6. Master index

At the top of the Markdown catalogue, provide one compact table with output counts, main files, and completeness for:

- models and fold runs;
- OOF result sets;
- primary/secondary metrics and paired comparisons;
- concept/intervention/contribution/alpha results;
- Grad-CAM and undefined maps;
- spatial faithfulness;
- existing/planned tables and figures;
- qualitative cases;
- private and public artifacts.

Immediately after this compact index, embed or link the deterministic `RESULTS_ARTIFACTS_MASTER_TABLE` human-readable view so the user can move from high-level counts to every run/result/artifact without opening machine-oriented registry files.

## 7. Planned interfaces

```bash
python -m lidc_baseline.p10_catalogue build
python -m lidc_baseline.p10_catalogue verify
python -m lidc_baseline.p10_catalogue export-private
python -m lidc_baseline.p10_catalogue verify-report-inputs
```

The future implementation must derive the catalogue from verified frozen artifacts, generate all views atomically, and produce a manifest containing schema version, source hashes, row counts, view hashes, and the canonical registry digest.

## 8. Validation plan

The future catalogue implementation must test:

- exact expected cardinalities for training, OOF, comparisons, concepts, contributions, alpha, Grad-CAM, faithfulness, and cases;
- `73,724 = 66,769 + 6,955` at detail, pooled, and global levels;
- exact source file/field/hash binding;
- exact `RPT-TA02` case-level malignancy/concept prediction and target bindings, including categorical reader vote distributions and any modal display label;
- read-only full CT renderability only when frozen source, exact series/slice provenance, and exact ROI-to-full-volume mapping all verify;
- independent Prediction/WHERE/WHAT/WHY/HOW availability for `RPT-FA06`, with unpersisted case-level intervention evidence classified rather than recomputed;
- public root aliases versus private exact-path overlay;
- public privacy and private file permissions;
- deterministic registry/Markdown/CSV/XLSX parity;
- missing, extra, duplicate, stale, and tampered records;
- controlled unavailable states without recomputation;
- no P5-P9 source-manifest change;
- no unclassified scientifically important result.

## 9. Gate after catalogue implementation

Even after both plans are approved, only the catalogue may be implemented first. After the catalogue and all validations are complete:

1. stop without modifying or regenerating reports;
2. report the generated catalogue paths, SHA-256 values, counts, gaps, and verification evidence;
3. wait for explicit user approval of the actual catalogue;
4. begin report revision only after that additional approval.

P10 remains incomplete throughout this process, and P11 remains undefined and forbidden.
