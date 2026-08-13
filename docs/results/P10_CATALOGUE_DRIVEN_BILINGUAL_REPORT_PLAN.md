# P10 Catalogue-Driven Bilingual Final Report Revision Plan

## 1. Independent approval gate

```text
P10_REPORT_PLAN_APPROVED=0
IMPLEMENTATION_AUTHORIZED=0
```

This document is a plan only. Creating or approving it does not revise the report generator, render a figure, rebuild a PDF, select a new case, or authorize any new scientific computation.

Approval rules:

- User approval must explicitly name `P10_CATALOGUE_DRIVEN_BILINGUAL_REPORT_PLAN.md`.
- Approval is bound to the exact file SHA-256 reported to the user.
- Any later content change invalidates only this plan's approval and resets `P10_REPORT_PLAN_APPROVED=0`.
- No implementation may begin until both this plan and `RESULTS_CATALOGUE_PLAN.md` are explicitly approved at their reported SHA-256 values.
- After both plans are approved, the Results Catalogue must be implemented, verified, delivered, and separately approved before any report revision begins.
- Approval of this plan is therefore approval of a future design, not approval to bypass the Catalogue gate.

## 2. Purpose, audience, and scientific boundary

Rebuild the P10 bilingual deliverables as continuous academic reports for supervisors and reviewers. The reports must explain one coherent story:

```text
Prediction -> WHERE -> WHAT -> WHY -> HOW -> Integrated interpretation
```

The revision replaces the current audit-style, one-section-per-page layout with evidence-led prose in which every scientific table or figure appears near its first citation and is followed by substantive interpretation.

Hard boundaries:

- Read P5-P9 frozen evidence only.
- Do not train, create an optimizer, run test inference, run a model forward pass, or start a new scientific job.
- Do not reselect checkpoints or rerun any committed test transaction.
- Do not modify P4 splits/initializations or P5-P9 checkpoints, histories, predictions, metrics, evaluations, OOF rows, interventions, Grad-CAM maps, occlusion rows, or faithfulness artifacts.
- Do not recompute bootstrap draws, intervention results, contributions, Grad-CAM, occlusion, faithfulness, or any scientific estimate merely to improve presentation.
- Do not introduce Guided Backprop, Guided Grad-CAM, SHAP, LIME, Integrated Gradients, a new Grad-CAM variant, or any other XAI method not frozen in P5-P9.
- Display-only transformation of frozen evidence is allowed only when registered by the Catalogue and traceable to its source hash.
- P11 is undefined and must not be created or started.

## 3. Mandatory Catalogue dependency

The report generator is `CATALOGUE_DRIVEN`, not merely Catalogue-informed. It must not discover results by scanning ad hoc paths, rely on author memory, copy from a previous PDF, use a hard-coded stale result list, or construct scientific inputs from unregistered paths. Its only scientific inventory is the approved Catalogue registry and its approved report mappings.

Required inputs:

```text
docs/results/results_catalogue_registry.json
docs/results/catalogue_manifest.json
docs/results/catalogue_to_report_plan.csv
reports/baseline_v2/p10/manifests/
```

The build must fail closed when:

- the Catalogue is absent, unapproved, stale, or hash-mismatched;
- any required section/table/figure item is missing or duplicated;
- a source artifact or field hash differs from the Catalogue;
- a scientifically important registered item is omitted without an approved omission reason;
- a report attempts to use an item classified as unavailable or as requiring new scientific computation;
- public output requests private-only evidence;
- reverse traceability is incomplete.

Evidence precedence is fixed:

```text
verified frozen P5-P9 artifact
-> approved Catalogue registry item
-> approved report section manifest
-> shared manuscript data model
-> table/figure/caption/prose
-> English/Chinese renderers
```

## 4. Deliverables

### 4.1 Mandatory public Git deliverables

1. English complete technical research report, Markdown and PDF.
2. Chinese complete technical research report, Markdown and PDF.

The two Markdown files and two corresponding public PDFs share the same structured evidence model. Public support files include aggregate figures, table CSVs, bilingual data dictionaries, source manifests, and reverse-traceability records.

English and Chinese short reports remain supported only as `OPTIONAL_LATER_DERIVATIVE`. They must not block P10 completion, enter the mandatory approval gate, or enter the mandatory PDF QA gate. If a short derivative is generated later under separate approval, it must use the same approved Catalogue/manuscript model and receive its own independent validation.

### 4.2 Private Mac-only deliverables

1. English qualitative appendix, Markdown and PDF.
2. Chinese qualitative appendix, Markdown and PDF.
3. English technical report plus English appendix combined PDF.
4. Chinese technical report plus Chinese appendix combined PDF.
5. Full-resolution panel sources and the restricted private case index.

Private deliverables never enter GitHub or Git LFS. They use opaque `CASE-####` labels in all human-readable output.

### 4.3 Mandatory PDF set subject to page-level QA

Exactly six PDFs require the complete P10 PDF quality gate:

```text
technical_en.pdf
technical_zh.pdf
qualitative_appendix_en.pdf
qualitative_appendix_zh.pdf
technical_en_with_appendix.pdf
technical_zh_with_appendix.pdf
```

## 5. Format and length

- A4, single-column, journal-style layout for adviser/reviewer reading.
- Flow-based ReportLab Platypus composition, not fixed one-section-per-page canvases.
- Cover placeholders: Author, Affiliation, Supervisor, Date.
- Technical main body editorial target is approximately 25-35 pages and 8,000-12,000 English words, excluding references, the public reproducibility appendix where appropriate, and the private qualitative appendix.
- Page and word ranges are guidance, not scientific hard gates. A complete, clear report must not fail solely because it is 24 or 36 pages, and prose must never be padded to reach a target.
- Scientific completeness, clarity, readability, and integration of evidence take precedence over page count.
- Any future implementation must place this revised nonblocking editorial rule in the separately reviewed report-revision execution supplement; this plan does not modify the currently frozen P10 report/archive supplement.
- Optional short-report derivatives, if separately approved later, may target 8-12 pages and approximately 2,500-4,000 English words without becoming P10 completion requirements.
- Chinese text is paragraph-level semantic correspondence, not mechanical word-count padding.
- Long tables may use landscape pages or repeating-header `LongTable` layouts.
- Captions stay bound to tables/figures; `KeepWithNext` and `KeepTogether` keep evidence near its analysis.

## 6. Technical-report structure

### Front matter

- Title page.
- Abstract, editorial target 250-350 words.
- Keywords.
- Table of contents.
- List of tables.
- List of figures.
- Abbreviation list.

### 1. Introduction

Explain clinical context, the prediction-versus-explanation gap, the evaluation objective, contributions, and the WHERE-WHAT-WHY-HOW framework. Editorial target: approximately 2-3 pages. Do not add a table merely for appearance.

### 2. Related Work

Cover Black-box models, concept bottleneck models, mixed-type CEMs, local-expert GAMs, Grad-CAM, occlusion faithfulness, concept interventions, and lung-nodule concept explanation. Editorial target: approximately 2-3 pages. Include the directly relevant Dumaev et al. concept-based malignancy paper. Use verified primary-source metadata in complete IEEE numeric format and clearly distinguish prior work from this project’s Mixed-type CEM and Learned-softmax GAM.

### 3. Dataset and Preprocessing

Describe only this project's frozen LIDC-IDRI cohort construction, radiologist-assessed malignancy target, eight concepts, exclusion logic, 64-cubed ROI, normalization, and patient-grouped folds. Editorial target: approximately 2-3 pages. Reference-paper cohort sizes such as 2,651 belong only in Related Work and must never be presented as the starting point from which this project's 2,633 nodules were obtained.

State explicitly:

```text
Malignancy is the downstream radiologist-assessed 1-5 target and is NOT one of the eight bottleneck concepts.
```

The eight concepts are exactly subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, and texture.

Also explain:

```text
ROI = 64 x 64 x 64 preprocessed local pulmonary-nodule patch used as model input.
It is not a complete axial CT slice.
Because it has been cropped and resampled, an ROI slice can appear lower-resolution than the original CT image.
```

### 4. Methods

Describe the four architectures, objectives, contribution semantics, learned GAM alpha, Grad-CAM target/formula, occlusion protocol, interventions, and statistical definitions. Editorial target: approximately 5-7 pages. Every architecture schematic and relevant caption must keep malignancy outside the eight-concept bottleneck and show it only as the downstream target/output.

### 5. Experimental Setup

Describe five-fold grouping, frozen initialization, training/evaluation separation, exactly-once committed test evaluation, validation-only threshold selection, bootstrap, resources, determinism, and execution provenance without turning the section into a scheduler log. Editorial target: approximately 2-3 pages.

### 6. Results

The order is mandatory:

1. Prediction: primary regression, paired Delta-MAE, extreme AUROC/AUPRC, paired Delta-AUROC.
2. WHERE: Grad-CAM accounting, undefined maps, spatial faithfulness, and private qualitative evidence references.
3. WHAT: continuous and categorical concept fidelity.
4. WHY: centered contributions, empirical OOF contribution profiles, learned GAM alpha, and case-level contribution evidence.
5. HOW: `k=0..8` interventions, `iMAE/Delta_iMAE`, `iAUC/Delta_iAUC`, random/permutation and error-first orderings.
6. Integrated interpretation: connect prediction quality and all four explanatory layers into one evidence chain.

Results have an approximate editorial target of 6-10 pages, determined by scientific completeness rather than padding.

### 7. Discussion

Interpret the principal findings, negative and unexpected results, comparison with related work, and implications. Editorial target: approximately 3-5 pages. Do not restate the Results section as a metric list.

### 8. Limitations

Use full prose to separate methodological and interpretive limitations: radiologist labels rather than pathology, no external validation, intervention semantics, Grad-CAM limitations, concentrated undefined maps, and unavailable pre-ReLU/gradient decomposition.

### 9. Conclusion

Provide a concise multi-paragraph conclusion without an audit-style checklist or a forced table.

### Public reproducibility appendix

Place execution provenance, source hashes, storage, recovery classification, privacy and governance evidence here so they do not displace the scientific story.

## 7. Results writing rules

Every Results subsection must:

1. answer `What was measured?` by stating the scientific question and registered method/metric;
2. answer `What did we observe?` with the relevant Catalogue-bound evidence;
3. answer `What does this mean for the research question?` without exceeding the registered interpretation boundary;
4. cite its Catalogue items and source evidence;
5. place the first relevant table or figure near its first textual reference;
6. provide at least three substantive paragraphs in the technical report;
7. identify negative or unexpected evidence;
8. state the limitation or interpretation boundary.

The local narrative sequence is a hard requirement:

```text
scientific question
-> method/metric
-> nearby table/figure
-> observed result
-> interpretation
-> limitation
```

Forbidden patterns:

- a table or figure without analysis;
- generic captions that do not state what the reader should inspect;
- a figure dump at the end of the report;
- repeated disclaimer paragraphs used to fill pages;
- calling a train-fold centering constant importance;
- calling an empirical contribution profile a causal shape function;
- using output sensitivity alone as evidence of prediction worsening.

Introduction, Discussion, and Conclusion do not require tables.

## 8. Planned public tables

All tables have stable IDs and must be registered in CAT-O before rendering.

| ID | Table | Scientific role |
|---|---|---|
| RPT-T01 | Related-work comparison | Compare task, concepts, spatial explanation, intervention and validation scope. |
| RPT-T02 | Our frozen cohort flow | Report only this project's frozen cohort construction, exclusions, 2,633 primary nodules, 868 patients, 1,073 extreme nodules and 578 extreme patients; never start from a reference-paper 2,651-nodule cohort. |
| RPT-T03 | Target and concept definitions | Define malignancy explicitly as the downstream radiologist-assessed 1-5 target, not a bottleneck concept; define the six continuous and two categorical bottleneck concepts and their encodings. |
| RPT-T04 | Four-model architecture comparison | Compare inputs, concept layer, decision layer, contribution and intervention semantics. |
| RPT-T05 | Frozen training configuration | Summarize registered model training settings and resources without scheduler-noise substitution. |
| RPT-T06 | Evaluation protocol | Summarize folds, metrics, bootstrap, threshold and faithfulness definitions. |
| RPT-T07 | Primary regression | Four-model point estimates and every existing 2,000 patient-cluster bootstrap CI. |
| RPT-T08 | Six paired Delta-MAE comparisons | Estimate, 95% CI, zero-crossing and directional interpretation. |
| RPT-T09 | Extreme-task performance | AUROC, AUPRC and registered CIs for all four models. |
| RPT-T10 | Six paired Delta-AUROC comparisons | Estimate, 95% CI, zero-crossing and direction. |
| RPT-T11 | Continuous concept metrics | Six concepts by three concept models with MAE, RMSE, Pearson, Spearman and N. |
| RPT-T12 | Categorical concept metrics | Two concepts by three concept models with CE, Brier, macro-F1, ties and N. |
| RPT-T13 | Grad-CAM accounting | Main Results uses model-level and important target-level concentrations, including the explicit CEM malignancy limitation; complete model x fold x target accounting remains in machine-readable CSV, Catalogue, and technical/reproducibility appendix. |
| RPT-T14 | Spatial faithfulness | Saliency, matched random, difference and win rate for output sensitivity and error increase. |
| RPT-T15 | Centered contribution summary | Report frozen OOF centered contributions using available mean absolute magnitude, 2.5th percentile, median, 97.5th percentile and sign/direction; keep train-fold centering constants in separate columns and never label them importance. |
| RPT-T16 | Fold-level learned GAM alpha | Concept x fold x expert weights and deviation from uniform initialization. |
| RPT-T17 | Intervention summary | `k=0`, selected intermediate k, `k=8`, random/permutation and error-first summaries. |
| RPT-T18 | WHERE-WHAT-WHY-HOW synthesis | Integrate predictive and explanatory evidence by model. |

Private Table `RPT-TA01` is the frozen 14-case index containing only opaque labels, model, role, target, ground truth, prediction, error and map status. Exact UID/patient mappings remain restricted to the private machine index.

## 9. Planned public figures

All figures have stable IDs and must be registered in CAT-P before rendering.

| ID | Figure | Scientific role |
|---|---|---|
| RPT-F01 | End-to-end evidence pipeline | Show Prediction, WHERE, WHAT, WHY and HOW in one analysis flow. |
| RPT-F02 | Cohort, preprocessing and five-fold flow | Explain sample construction and patient grouping. |
| RPT-F03 | Four architectures and interpretability interfaces | Compare Black-box, Standard CBM, Mixed-type CEM and Learned-softmax GAM. |
| RPT-F04 | Four-model MAE bootstrap intervals | Display primary pooled error with patient-cluster uncertainty. |
| RPT-F05 | Paired Delta-MAE forest plot | Show six registered comparisons and zero crossing. |
| RPT-F06 | Extreme AUROC/AUPRC and paired Delta-AUROC | Combine secondary performance and six paired comparisons without obscuring sign conventions. |
| RPT-F07 | Undefined Grad-CAM rate heatmap | Display model x fold x target concentration. |
| RPT-F08 | Spatial faithfulness dual panel | Compare saliency and matched random for output sensitivity and error increase separately. |
| RPT-F09 | Concept prediction performance | Use separate continuous and categorical panels with independent scales and metric semantics; never mix MAE/RMSE/Pearson/Spearman with CE/Brier/macro-F1 on one colour scale. |
| RPT-F10 | Empirical OOF contribution profiles | Show six continuous binned profiles and two categorical distributions for all concept models. |
| RPT-F11 | GAM alpha heatmap | Show fold x concept x expert learned weights. |
| RPT-F12 | Intervention curves | Show `k=0..8` iMAE/Delta_iMAE and iAUC/Delta_iAUC under registered orderings. |
| RPT-F13 | Integrated evidence synthesis | Link Prediction, WHERE, WHAT, WHY and HOW to the supported conclusion codes. |

Figure `RPT-F10` is explicitly descriptive: continuous profiles use frozen OOF points and preregistered bin summaries; categorical profiles use predicted-class distributions. It must be labelled “empirical OOF contribution profile” and never “global causal shape function.”

## 10. Planned private figures

The private qualitative appendix contains five registered paper-style figure types, all derived from frozen cases and existing artifacts. They must be designed for supervisors/reviewers, not presented as debug panels:

| ID | Private figure | Content |
|---|---|---|
| RPT-FA01 | Representative-case comparison | Median-error cases with full axial CT context, ROI box, zoomed ROI, available four-model task overlays and prediction/target evidence. |
| RPT-FA02 | Maximum-error failure comparison | Maximum-error cases with full CT/ROI context, prediction error and available spatial evidence. |
| RPT-FA03 | Concept contribution explanation | Full CT/ROI context plus malignancy prediction/target, concept prediction/GT and signed centered contribution bars. |
| RPT-FA04 | Intervention-worsening cases | Existing largest positive error-worsening evidence for each concept model, paired with CT/ROI context where frozen metadata permit. |
| RPT-FA05 | Undefined zero-map limitation | Genuine frozen all-zero post-ReLU map for a high-undefined target, with CT/ROI context and a valid comparison map where available. |

Case rules:

- Use the already frozen 14 cases; do not select or infer new cases during report rendering.
- A frozen case may display existing task maps from multiple models without increasing the number of cases.
- ROI-only panels must not be the primary or sole case presentation when registered frozen full-slice metadata are available.
- Use the Catalogue to determine availability of full CT, frozen z index, ROI bounding box, zoomed ROI, ROI overlay, full-slice reprojection and concept-specific targets. Do not scan ad hoc paths during rendering.
- Paper-style case figures prioritize human interpretability over implementation/debug convenience.
- Valid maps have a color bar; undefined maps remain visibly all zero.
- Do not infer the unavailable pre-ReLU, gradient, channel-weight, or exact-mechanism decomposition.
- Strip image metadata and expose no UID, patient key, absolute path or private approval record.

### 10.1 Required case-panel composition

For every selected case, render the following from frozen data when the Catalogue marks the component available:

**Panel A - full axial CT context**

- original complete axial CT slice at the registered frozen z index;
- pulmonary-window or registered grayscale display;
- clearly marked ROI bounding box.

**Panel B - zoomed ROI**

- zoomed local nodule ROI crop;
- registered grayscale/windowing;
- no misleading image enhancement.

**Panel C - ROI plus Grad-CAM**

- existing frozen Grad-CAM overlay;
- display-only normalization allowed;
- colour bar for valid maps.

**Panel D - full CT plus reprojected Grad-CAM**

- only when frozen spatial-transform metadata permit exact placement;
- place heatmap values only inside the mapped ROI region;
- never fabricate values outside the ROI.

If reprojection cannot be derived from frozen metadata, the panel is labelled `FULL_SLICE_REPROJECTION_NOT_AVAILABLE_FROM_FROZEN_DATA`; no new forward pass, registration, or scientific estimate is allowed.

### 10.2 Concept-specific maps and case explanations

For selected concept-model cases, include existing valid maps for malignancy, spiculation, margin, and texture where available. Additional concept maps may be shown only when scientifically useful, frozen, and Catalogue-registered. Undefined or unavailable maps are never treated as required colourful panels.

Case-level contribution figures must not show contribution bars alone. They pair:

```text
full CT / ROI context
+ malignancy prediction versus radiologist mean target
+ predicted concepts versus ground-truth reader targets
+ signed centered contribution bars
```

The figure must answer both `WHAT did the model predict?` and `WHY did those concepts produce this malignancy score?`

### 10.3 Undefined zero-map limitation figure

At least one dedicated figure uses a genuine frozen undefined map and labels it:

```text
Undefined - post-ReLU all-zero Grad-CAM
```

Where available, show full CT context, ROI, and a valid comparison map from another registered model/target. Never manufacture a colourful substitute or claim the exact mechanism is known when pre-ReLU/gradient decomposition was not persisted.

### 10.4 Display-only normalization and windowing

The scientific artifact remains unchanged:

```text
raw FP32 Grad-CAM map
-> display-only normalization
-> qualitative overlay
```

Every relevant caption states:

```text
Grad-CAM heatmaps were normalized only for visualization. Quantitative occlusion/faithfulness analysis used the original unnormalized FP32 maps.
```

CT grayscale/windowing may improve human readability without modifying frozen underlying values, and the applied policy must be registered in the Catalogue.

### 10.5 CT-context policy by figure type

CT context is required or preferred for qualitative CT/Grad-CAM cases, failures, undefined maps, case-level concept predictions, case-level contributions, and integrated Prediction-WHERE-WHAT-WHY-HOW examples.

CT context is not required and must not be added merely for decoration to aggregate MAE/CI, paired Delta-MAE, AUROC/AUPRC, paired Delta-AUROC, concept-metric summaries, intervention curves, GAM-alpha heatmaps, empirical contribution profiles, Grad-CAM accounting heatmaps, or aggregate spatial-faithfulness figures.

### 10.6 Public versus private figure policy

Public Git reports may contain aggregate tables, aggregate statistical plots, architecture/workflow diagrams, and deidentified numeric evidence. They must exclude raw CT, raw ROI, CT case panels, raw Grad-CAM volumes, private case mappings, UID, and patient keys.

Private reports/appendices may contain deidentified qualitative CT figures using opaque `CASE-####` labels only. All exported images have identifying metadata removed.

## 11. Shared manuscript data model

Build one immutable structured model containing:

- section and paragraph IDs;
- Catalogue result/artifact IDs;
- table and figure specifications;
- scientific conclusion codes;
- exact source artifact, field path and SHA-256;
- formatted values and uncertainty intervals;
- English and Chinese prose/caption/label templates;
- citation and cross-reference IDs;
- availability and privacy class;
- omission reason and approval reference.

Scientific numbers are formatted once from this model. Translation must not recompute, re-round, or manually retype numbers.

## 12. Section manifests and reverse traceability

Each major section has a machine-readable manifest under:

```text
reports/baseline_v2/p10/manifests/
```

Required fields:

```text
section_id
catalogue_registry_sha256
catalogue_items_required
catalogue_items_used
required_result_ids
required_artifact_ids
required_table_ids
tables_rendered
required_figure_ids
figures_rendered
private_cases_required
private_cases_rendered
conclusion_codes
omitted_catalogue_ids_and_reasons
omission_approval
privacy_scope
source_hashes
english_render_sha256
chinese_render_sha256
verification_status: PASS | BLOCKED
```

A required Catalogue item cannot be silently omitted. The verifier compares required versus used/rendered sets exactly and blocks until each omission has an explicit reason and approval reference.

Forward traceability:

```text
Catalogue item -> section/table/figure/caption/conclusion
```

Reverse traceability:

```text
report claim/table cell/plot series -> Catalogue item -> frozen source field -> source SHA
```

The final verifier must prove both directions for every scientific number and conclusion code.

## 13. Bilingual consistency

- English and Chinese use the identical section, table, figure, reference and conclusion-code IDs.
- Model names and abbreviations remain: Black-box, Standard CBM, Mixed-type CEM, Learned-softmax GAM.
- Ordinary axis labels, legends and explanatory annotations are translated; model names and canonical concept names follow the glossary.
- All numerical tokens, table cells, interval bounds, sample counts, fold counts, hashes, sign conventions and zero-crossing flags must match exactly.
- Shared figures use identical data and geometry; only labels/captions change language.
- References use the same numeric ordering and complete IEEE-style primary-source metadata in both languages.
- Chinese PDF fonts use embedded Songti SC Regular/Bold from the registered `Songti.ttc` subfonts, with font SHA recorded in the audit.
- Missing glyphs, substitution squares and unembedded fonts are blocking failures.

Mandatory statements in both languages:

- Primary scores are unclipped.
- LIDC malignancy is a radiologist assessment, not a pathology-confirmed diagnosis.
- Malignancy is the downstream radiologist-assessed 1-5 target and is NOT one of the eight bottleneck concepts.
- The system is not a clinical diagnostic product.
- CEM means this project’s mixed-type CEM.
- GAM means the preregistered learned-softmax local-expert design.
- The 6,955 undefined maps are confirmed post-ReLU all-zero maps; exact pre-ReLU/gradient decomposition was not persisted and cannot be over-interpreted without prohibited new forward passes.
- The observed concentration is reported with the registered `SYSTEMATIC_MODEL/TARGET_ISSUE` limitation label, not silently treated as an implementation error.

### 13.1 Reference and prior-work policy

The report requires verified primary-source citations for LIDC-IDRI, DenseNet, CBM, CEM, Grad-CAM, bootstrap/statistical methodology, Dumaev et al. concept-based pulmonary-nodule malignancy work, and relevant GAM/additive-model literature. Both languages use identical IEEE numeric ordering and verified DOI/arXiv metadata.

Prior work, this project’s Mixed-type CEM, and this project’s Learned-softmax GAM must be distinguished explicitly. Reference-paper tables, figures, cohort definitions, and scientific outputs may inspire layout or discussion but must not be copied or represented as this project’s evidence.

## 14. Planned build interfaces

Existing public command semantics remain available but become Catalogue-gated:

```bash
python -m lidc_baseline.p10_report verify-inputs
python -m lidc_baseline.p10_report build --variant short --language en
python -m lidc_baseline.p10_report build --variant short --language zh
python -m lidc_baseline.p10_report build --variant technical --language en
python -m lidc_baseline.p10_report build --variant technical --language zh
python -m lidc_baseline.p10_report build-private-appendix --language en
python -m lidc_baseline.p10_report build-private-appendix --language zh
python -m lidc_baseline.p10_report verify --scope all
python -m lidc_baseline.p10_audit build
```

The two `--variant short` commands are optional later derivatives and are outside the mandatory P10 completion/QA gate. No CLI flag may bypass the approved Catalogue SHA, approved report-plan SHA, section manifests, privacy gate, or reverse traceability.

## 15. Validation and visual QA

### 15.1 Scientific and traceability checks

- P5-P9 source manifest and scientific artifact hashes are unchanged.
- All values agree with approved Catalogue items and section manifests.
- `73,724 = 66,769 + 6,955` and every model x fold x target breakdown agree exactly.
- Four-model metrics, every available bootstrap CI, all six paired comparisons, eight concepts, GAM alpha, intervention and both faithfulness quantities agree with frozen evidence.
- Every table row, plot series, caption claim and conclusion code has reverse traceability.
- Every required Catalogue item is used or has an approved omission reason.
- No report output creates a new scientific estimate.

### 15.2 Narrative and layout checks

- All Results layers appear in Prediction-WHERE-WHAT-WHY-HOW order.
- Every table/figure is cited and appears near substantive analysis.
- No end-of-document figure dump exists.
- Technical Results subsections contain at least three substantive paragraphs.
- Page and word ranges are evaluated as editorial diagnostics, not completion blockers; scientific completeness and clarity take precedence.
- Duplicate template paragraphs, repeated disclaimers and page-count padding fail validation.

### 15.3 Privacy checks

- Public files reject UID, patient key, absolute path, raw maps, CT panels, private index and approval records.
- Private case selection matches the frozen 14-case index and records `model_forward=false`.
- Private panels use opaque labels and scrubbed metadata.
- Public/private Git candidates are checked against exact whitelists and file-size gates.

### 15.4 Bilingual checks

- Section/table/figure/reference IDs are one-to-one.
- Numerical tokens, table cells, CIs, zero-crossing flags and conclusion codes are identical.
- Chinese labels are translated according to the glossary.
- English and Chinese use identical cases, slices, maps, panel layouts and chart geometry.

### 15.5 PDF checks

For all six mandatory PDFs:

- render every page with Poppler at 150 DPI;
- verify page count, table of contents, extractable text, metadata, references, table/figure numbering and fonts with pypdf/pdfplumber;
- inspect contact sheets page by page;
- inspect wide tables, Grad-CAM panels, captions and dense pages at original render resolution;
- fail on clipping, overlap, missing headers/footers, orphan captions, broken titles, unreadable legend, incorrect image resolution, font substitution or blank/missing pages.

Manual visual-review evidence must identify reviewer, timestamp, PDF SHA, rendered-page manifest SHA and PASS/FAIL. The audit must never infer a PASS merely because rendering completed.

## 16. Plan-level consistency checks before implementation

Before any future implementation begins, search both full plans and block on contradictions involving:

- mandatory versus `OPTIONAL_LATER_DERIVATIVE` short reports;
- exactly six mandatory PDFs;
- approximately 25-35 pages as an editorial target rather than a scientific hard gate;
- this project's frozen cohort versus any reference-paper cohort;
- malignancy as downstream target versus the eight bottleneck concepts;
- continuous versus categorical metric semantics and scales;
- centering constants versus contribution importance;
- public versus private images;
- ROI patch versus full CT slice;
- Grad-CAM display normalization versus quantitative raw FP32 maps;
- allowed display transformations versus prohibited scientific recomputation;
- Catalogue-driven report generation and reverse traceability;
- P10 completion and approval gates.

Any requirement not fully represented is reported as `BLOCKED_NOT_FULLY_IN_PLAN`; it must not be silently claimed complete.

## 17. Implementation sequence after approvals

This plan does not authorize implementation. The future sequence is fixed:

1. User explicitly approves both plan files at their exact SHA-256 values.
2. Implement and verify the Results Catalogue only.
3. Stop and deliver the actual Catalogue, manifests, counts, gaps, SHA values and verification evidence.
4. Wait for explicit user approval of the actual Catalogue.
5. Only then implement the Catalogue-driven manuscript model, section manifests, report generator and reverse traceability.
6. Rebuild the two mandatory public bilingual technical reports, two private bilingual appendices and two combined PDFs. Optional short derivatives are not part of this gate.
7. Complete scientific, bilingual, privacy, layout and six-PDF visual QA.
8. Run full tests, frozen checks, Phase Compliance Review and Status Synchronization Review.
9. Transition P10 only to `AWAITING_USER_APPROVAL`.
10. Wait for final user confirmation before `COMPLETED`, fast-forward merge, main-branch retest and GitHub push.

No P11 is created at any point.
