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
- Do not modify P4 splits/initializations or P5-P9 checkpoints, histories, predictions, metrics, evaluations, OOF rows, interventions, Grad-CAM maps, occlusion rows, or faithfulness artifacts.
- Do not recompute bootstrap draws or scientific estimates.
- Display-only transformation of frozen evidence is allowed only when registered by the Catalogue and traceable to its source hash.
- P11 is undefined and must not be created or started.

## 3. Mandatory Catalogue dependency

The report generator must not discover results by scanning ad hoc paths or rely on author memory. Its only scientific inventory is the approved Catalogue registry and its approved report mappings.

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
-> English/Chinese renderers
```

## 4. Deliverables

### 4.1 Public Git deliverables

Four canonical public bilingual reports:

1. English short report, Markdown and PDF.
2. Chinese short report, Markdown and PDF.
3. English technical report, Markdown and PDF.
4. Chinese technical report, Markdown and PDF.

The four Markdown files and four corresponding public PDFs share the same structured evidence model. Public support files include aggregate figures, table CSVs, bilingual data dictionaries, source manifests, and reverse-traceability records.

### 4.2 Private Mac-only deliverables

1. English qualitative appendix, Markdown and PDF.
2. Chinese qualitative appendix, Markdown and PDF.
3. English technical report plus English appendix combined PDF.
4. Chinese technical report plus Chinese appendix combined PDF.
5. Full-resolution panel sources and the restricted private case index.

Private deliverables never enter GitHub or Git LFS. They use opaque `CASE-####` labels in all human-readable output.

### 4.3 PDF set subject to page-level QA

Exactly eight PDFs require the complete PDF quality gate:

```text
short_en.pdf
short_zh.pdf
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
- Technical main body must remain within the frozen 25-35 page gate, with an editorial target of approximately 30-35 pages and 8,000-12,000 English words. References and private appendix pages do not count toward this body target.
- Short-report target: 8-12 pages and approximately 2,500-4,000 English words.
- Chinese text is paragraph-level semantic correspondence, not mechanical word-count padding.
- Long tables may use landscape pages or repeating-header `LongTable` layouts.
- Captions stay bound to tables/figures; `KeepWithNext` and `KeepTogether` keep evidence near its analysis.

## 6. Technical-report structure

### Front matter

- Title page.
- Abstract.
- Keywords.
- Table of contents.
- List of tables.
- List of figures.
- Abbreviation list.

### 1. Introduction

Explain clinical context, the prediction-versus-explanation gap, the evaluation objective, contributions, and the WHERE-WHAT-WHY-HOW framework. Do not add a table merely for appearance.

### 2. Related Work

Cover Black-box models, concept bottleneck models, mixed-type CEMs, local-expert GAMs, Grad-CAM, occlusion faithfulness, concept interventions, and lung-nodule concept explanation. Include the directly relevant Dumaev et al. concept-based malignancy paper. Use verified primary-source metadata in complete IEEE numeric format.

### 3. Dataset and Preprocessing

Describe cohort flow, radiologist-assessed malignancy target, eight concepts, exclusion logic, 64-cubed ROI, normalization, and patient-grouped folds.

### 4. Methods

Describe the four architectures, objectives, contribution semantics, learned GAM alpha, Grad-CAM target/formula, occlusion protocol, interventions, and statistical definitions.

### 5. Experimental Setup

Describe five-fold grouping, frozen initialization, training/evaluation separation, exactly-once committed test evaluation, validation-only threshold selection, bootstrap, resources, determinism, and execution provenance without turning the section into a scheduler log.

### 6. Results

The order is mandatory:

1. Prediction: primary regression, paired Delta-MAE, extreme AUROC/AUPRC, paired Delta-AUROC.
2. WHERE: Grad-CAM accounting, undefined maps, spatial faithfulness, and private qualitative evidence references.
3. WHAT: continuous and categorical concept fidelity.
4. WHY: centered contributions, empirical OOF contribution profiles, learned GAM alpha, and case-level contribution evidence.
5. HOW: `k=0..8` interventions, `iMAE/Delta_iMAE`, `iAUC/Delta_iAUC`, random/permutation and error-first orderings.
6. Integrated interpretation: connect prediction quality and all four explanatory layers into one evidence chain.

### 7. Discussion

Interpret the principal findings, negative and unexpected results, comparison with related work, and implications. Do not restate the Results section as a metric list.

### 8. Limitations

Separate methodological and interpretive limitations: radiologist labels rather than pathology, no external validation, intervention semantics, Grad-CAM limitations, concentrated undefined maps, and unavailable pre-ReLU/gradient decomposition.

### 9. Conclusion

Provide a concise multi-paragraph conclusion without an audit-style checklist or a forced table.

### Public reproducibility appendix

Place execution provenance, source hashes, storage, recovery classification, privacy and governance evidence here so they do not displace the scientific story.

## 7. Results writing rules

Every Results subsection must:

1. begin with the scientific question and registered metric;
2. cite its Catalogue items and source evidence;
3. place the first relevant table or figure near its first textual reference;
4. provide at least three substantive paragraphs in the technical report;
5. explain the main numerical finding;
6. explain its scientific meaning;
7. identify negative or unexpected evidence;
8. state the interpretation boundary.

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
| RPT-T02 | Cohort flow | Report reference cohort, exclusions, 2,633 primary nodules, 868 patients, 1,073 extreme nodules and 578 extreme patients. |
| RPT-T03 | Target and concept definitions | Define malignancy, six continuous concepts, two categorical concepts and encodings. |
| RPT-T04 | Four-model architecture comparison | Compare inputs, concept layer, decision layer, contribution and intervention semantics. |
| RPT-T05 | Frozen training configuration | Summarize registered model training settings and resources without scheduler-noise substitution. |
| RPT-T06 | Evaluation protocol | Summarize folds, metrics, bootstrap, threshold and faithfulness definitions. |
| RPT-T07 | Primary regression | Four-model point estimates and every existing 2,000 patient-cluster bootstrap CI. |
| RPT-T08 | Six paired Delta-MAE comparisons | Estimate, 95% CI, zero-crossing and directional interpretation. |
| RPT-T09 | Extreme-task performance | AUROC, AUPRC and registered CIs for all four models. |
| RPT-T10 | Six paired Delta-AUROC comparisons | Estimate, 95% CI, zero-crossing and direction. |
| RPT-T11 | Continuous concept metrics | Six concepts by three concept models with MAE, RMSE, Pearson, Spearman and N. |
| RPT-T12 | Categorical concept metrics | Two concepts by three concept models with CE, Brier, macro-F1, ties and N. |
| RPT-T13 | Grad-CAM accounting | Model x fold x target requested, valid, undefined and rate, with pooled/global totals. |
| RPT-T14 | Spatial faithfulness | Saliency, matched random, difference and win rate for output sensitivity and error increase. |
| RPT-T15 | Centered contribution summary | Principal positive/negative centered contributions with magnitude and sign semantics. |
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
| RPT-F09 | Concept prediction performance | Present continuous and categorical metrics in coordinated panels. |
| RPT-F10 | Empirical OOF contribution profiles | Show six continuous binned profiles and two categorical distributions for all concept models. |
| RPT-F11 | GAM alpha heatmap | Show fold x concept x expert learned weights. |
| RPT-F12 | Intervention curves | Show `k=0..8` iMAE/Delta_iMAE and iAUC/Delta_iAUC under registered orderings. |
| RPT-F13 | Integrated evidence synthesis | Link Prediction, WHERE, WHAT, WHY and HOW to the supported conclusion codes. |

Figure `RPT-F10` is explicitly descriptive: continuous profiles use frozen OOF points and preregistered bin summaries; categorical profiles use predicted-class distributions. It must be labelled “empirical OOF contribution profile” and never “global causal shape function.”

## 10. Planned private figures

The private qualitative appendix contains five registered figure types, all derived from frozen cases and existing artifacts:

| ID | Private figure | Content |
|---|---|---|
| RPT-FA01 | Representative-case comparison | Median-error cases with ROI, four-model existing task maps and predictions where available. |
| RPT-FA02 | Maximum-error failure comparison | Maximum-error cases with prediction error and spatial evidence. |
| RPT-FA03 | Concept contribution explanation | GT/predicted concepts and signed contribution bars for concept-model cases. |
| RPT-FA04 | Intervention-worsening cases | Existing largest positive error-worsening evidence for each concept model. |
| RPT-FA05 | Undefined zero-map cases | Existing all-zero post-ReLU maps for the highest-undefined-rate target of each concept model. |

Case rules:

- Use the already frozen 14 cases; do not select or infer new cases during report rendering.
- A frozen case may display existing task maps from multiple models without increasing the number of cases.
- Use ROI source, ROI plus Grad-CAM overlay, and raw post-ReLU map where registered and available.
- Overlay normalization applies only to an in-memory display copy; raw FP32 maps remain unchanged.
- Valid maps have a color bar; undefined maps remain visibly all zero.
- Do not infer the unavailable pre-ReLU, gradient, channel-weight, or exact-mechanism decomposition.
- Strip image metadata and expose no UID, patient key, absolute path or private approval record.

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
required_result_ids
required_artifact_ids
required_table_ids
required_figure_ids
conclusion_codes
omitted_catalogue_ids_and_reasons
privacy_scope
source_hashes
english_render_sha256
chinese_render_sha256
verification_status
```

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
- The system is not a clinical diagnostic product.
- CEM means this project’s mixed-type CEM.
- GAM means the preregistered learned-softmax local-expert design.
- The 6,955 undefined maps are confirmed post-ReLU all-zero maps; exact pre-ReLU/gradient decomposition was not persisted and cannot be over-interpreted without prohibited new forward passes.
- The observed concentration is reported with the registered `SYSTEMATIC_MODEL/TARGET_ISSUE` limitation label, not silently treated as an implementation error.

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

No CLI flag may bypass the approved Catalogue SHA, approved report-plan SHA, section manifests, privacy gate, or reverse traceability.

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
- Technical English main body is approximately 8,000-12,000 words; short English is approximately 2,500-4,000 words.
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

For all eight PDFs:

- render every page with Poppler at 150 DPI;
- verify page count, table of contents, extractable text, metadata, references, table/figure numbering and fonts with pypdf/pdfplumber;
- inspect contact sheets page by page;
- inspect wide tables, Grad-CAM panels, captions and dense pages at original render resolution;
- fail on clipping, overlap, missing headers/footers, orphan captions, broken titles, unreadable legend, incorrect image resolution, font substitution or blank/missing pages.

Manual visual-review evidence must identify reviewer, timestamp, PDF SHA, rendered-page manifest SHA and PASS/FAIL. The audit must never infer a PASS merely because rendering completed.

## 16. Implementation sequence after approvals

This plan does not authorize implementation. The future sequence is fixed:

1. User explicitly approves both plan files at their exact SHA-256 values.
2. Implement and verify the Results Catalogue only.
3. Stop and deliver the actual Catalogue, manifests, counts, gaps, SHA values and verification evidence.
4. Wait for explicit user approval of the actual Catalogue.
5. Only then implement the Catalogue-driven manuscript model, section manifests, report generator and reverse traceability.
6. Rebuild the four public bilingual reports, two private bilingual appendices and two combined PDFs.
7. Complete scientific, bilingual, privacy, layout and eight-PDF visual QA.
8. Run full tests, frozen checks, Phase Compliance Review and Status Synchronization Review.
9. Transition P10 only to `AWAITING_USER_APPROVAL`.
10. Wait for final user confirmation before `COMPLETED`, fast-forward merge, main-branch retest and GitHub push.

No P11 is created at any point.
