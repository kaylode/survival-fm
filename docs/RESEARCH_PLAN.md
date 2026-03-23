# SurvPFN Research Plan
**Project:** Foundation Model Embeddings for Clinical Survival Analysis
**Target Deadline:** 2026-06-23 (CHIL 2026 / MLHC 2026)
**Last Updated:** 2026-03-23

---

## 1. Core Research Question

**Can tabular foundation model embeddings (TabPFN, SAINT) significantly improve survival analysis performance on small-to-medium clinical datasets, relative to classical feature inputs, when paired with purpose-built survival heads under a rigorous competing-risk evaluation protocol?**

More precisely: we test whether frozen or lightly fine-tuned in-context representations from TabPFN v2 provide a strictly superior inductive bias for survival tasks compared to raw features fed into the same survival architectures (DeepSurv, DeepHit, DSM), and whether the gain is consistent across two independent cohorts (Sirbu n=8,065; URRAH n=27,078) and four clinical endpoints (total mortality, CVD mortality, MI, stroke).

---

## 2. Candidate Approaches

Ranked by (novelty, feasibility, expected gain):

### Approach A — TabPFN Embedding + Survival Head (PRIMARY)
**Novelty: High | Feasibility: High | Expected Gain: Moderate–High**

Pipeline:
```
Raw clinical features
    → TabPFN v2 (frozen, n_fold K-fold embedding extraction)
    → Embedding vector (dim ~512)
    → Survival Head (DeepSurv / DeepHit / DSM)
    → Predicted survival function S(t|x)
```

Rationale:
- TabPFN v2's in-context learning captures non-linear feature interactions without gradient steps
- K-fold embedding extraction (as in TabPFNEmbedding, n_fold >= 2) avoids in-sample contamination and produces richer train-set representations
- Survival head is trained end-to-end on embeddings; foundation model weights remain frozen
- Directly addresses the dataset size regime (8K–27K samples) where TabPFN excels

Implementation status: `survpfn/embedding_cox.py` provides the embedding+CoxPH pipeline. DeepHit and DSM heads need to be connected to the same embedding extractor.

Key decisions:
- Use `TabPFNClassifier` for event prediction as the proxy task during embedding extraction (binary: event vs. censored)
- Compare n_fold in {0, 3, 5, 10} to quantify embedding quality vs. compute
- Evaluate both frozen and 1-layer fine-tuned embeddings

### Approach B — End-to-End TabPFN Fine-Tuning with Survival Loss (SECONDARY)
**Novelty: Very High | Feasibility: Moderate | Expected Gain: High if it works**

Pipeline:
```
Raw clinical features
    → TabPFN v2 (unfrozen, last N transformer layers)
    → Survival Head
    → Differentiable survival loss (DeepHit / DSM)
    → Backprop through survival loss into transformer
```

Rationale:
- Fine-tuning the foundation model with a survival-specific objective is unexplored in the literature
- Requires modifying TabPFN's forward pass to be differentiable with respect to survival targets (currently designed for classification)
- High risk: TabPFN uses in-context learning (the "training set" is part of the forward pass), so standard mini-batch fine-tuning is non-trivial

Mitigation: If full fine-tuning is infeasible in the timeline, use LoRA-style adapter layers inserted between transformer blocks and trained with survival loss while the base model is frozen. This reduces Approach B to an adapter variant of Approach A.

### Approach D — Retrieval-Augmented TabPFN for Survival (EXPLORATORY)
**Novelty: High | Feasibility: Moderate | Expected Gain: Moderate**

Pipeline:
```
For each test point x_i:
    → Compute cosine similarity to all training points (on raw / scaled features)
    → Retrieve top-K nearest training neighbours as TabPFN context
    → Run TabPFN forward pass with retrieved context only
    → Extract embedding → Survival Head
```

Rationale:
- Standard TabPFN uses the entire training set as context; for large cohorts (URRAH n=27K) this
  exceeds the ~10K context-window limit. Retrieval selects the most semantically relevant context.
- Hypothesis: a concentrated neighbourhood context improves embedding quality vs. a random
  sub-sample or full-context, especially for rare-event subgroups.
- Implemented in `experiments/tabpfn_combos.py` as `tabpfn_retrieval_k10` and
  `tabpfn_retrieval_k50` embedding variants.

Key hyperparameters: K ∈ {10, 50}; retrieval metric = cosine similarity on StandardScaler-normalised features.

Timeline: Retrieval results available as part of the 6×6 combo grid (Week 2). Analysis of K sensitivity deferred to Month 2 ablations.

### Approach C — TabPFN In-Context Survival Prediction via Discretization (EXPLORATORY)
**Novelty: Very High | Feasibility: Low–Moderate | Expected Gain: Unknown**

Pipeline:
```
Discretize time into K bins → Convert to K-class classification problem
    → Feed (X_train, time_bin_labels) as context into TabPFN
    → Query on X_test → Predicted bin probabilities = discrete survival function
    → Aggregate to CIF / cumulative hazard
```

Rationale:
- TabPFN's in-context learning can directly be repurposed for survival by treating discretized time-to-event as a multi-class target
- Competing risks map naturally to multi-class (event type × time bin) or cause-specific models
- No training required at all — pure ICL inference

Risks: TabPFN's context window limits (currently ~10K rows, 500 features), discretization resolution vs. accuracy, and whether soft probability outputs calibrate well for survival estimation.

Timeline note: Approach C is an ablation/exploratory experiment, not the primary paper claim. Allocate 2 weeks in Month 2 if Approach A results are strong.

---

## 2b. Experimental Matrix — TabPFN × Survival Head (6×6 Grid)

Implemented in `experiments/tabpfn_combos.py`. Run with:
```bash
uv run python experiments/tabpfn_combos.py --dataset GBSG --folds 5
uv run python experiments/tabpfn_combos.py --dataset sirbu --folds 5
```

### Embedding Variants (X axis)

| ID | Variant | Description |
|----|---------|-------------|
| E1 | `raw` | No TabPFN; raw scaled features (baseline) |
| E2 | `tabpfn_vanilla` | TabPFN, n_fold=0 (existing embedding_cox.py approach) |
| E3 | `tabpfn_kfold3` | TabPFN, n_fold=3 out-of-fold embeddings |
| E4 | `tabpfn_kfold5` | TabPFN, n_fold=5 out-of-fold embeddings |
| E5 | `tabpfn_retrieval_k10` | TabPFN, top-10 cosine-similar train neighbours as context |
| E6 | `tabpfn_retrieval_k50` | TabPFN, top-50 cosine-similar train neighbours as context |

### Survival Head Variants (Y axis)

| ID | Variant | Description |
|----|---------|-------------|
| H1 | `cox` | Cox Proportional Hazards (lifelines) |
| H2 | `deephit` | DeepHit discrete-time (pycox DeepHitSingle) |
| H3 | `deephit_single` | DeepHit single-risk (alias for H2) |
| H4 | `pchazard` | PC-Hazard / Logistic-Hazard (pycox) |
| H5 | `mtlr` | MTLR (pycox, if available) |
| H6 | `rsf` | Random Survival Forest (sksurv, non-deep baseline) |

### Datasets in scope

- **Private:** Sirbu (n≈6,500), URRAH (n≈24,000)
- **External benchmarks:** GBSG (n≈2,200), METABRIC (n≈1,900), SUPPORT2 (n≈9,000) — loaded via pycox built-ins

### Expected output

`results/tabpfn_combos_<dataset>.csv` — one row per (embedding, head, fold) with columns:
`dataset`, `embedding`, `head`, `fold`, `c_index`, `status`.

---

## 2c. Deferred / Future Work — Meta-Training TabPFN

**Out of scope for the current 3-month submission cycle.**

Meta-training TabPFN on survival data (i.e., fine-tuning the prior-fitted transformer on a corpus
of synthetic or real survival datasets so that its in-context learning is specialised for censored
outcomes) is a promising but high-effort direction:

- Requires modifying TabPFN's meta-training loop to generate synthetic survival datasets with
  censoring mechanisms (random, administrative, informative).
- Training infrastructure: large GPU cluster, multi-day run.
- Unclear if the standard TabPFN architecture supports arbitrary real-valued targets with
  censoring masks without significant re-engineering.

**Recommendation:** Treat this as a 6–12 month follow-up project after the primary paper is
submitted. If Approach A + retrieval show strong gains, the meta-training direction becomes a
compelling ICML/NeurIPS submission in its own right.

---

## 3. Three-Month Timeline

### Month 1: Foundation (2026-03-23 → 2026-04-22)

**Week 1 (by 2026-03-30) — Infrastructure & Bug Fixes**
- [ ] Code Reviewer: Audit and fix data leakage in `preprocessing.py` (outcome-correlated columns in feature matrix)
- [ ] Code Reviewer: Replace aggressive `dropna()` with `IterativeImputer` or `MissForest` for missing features; keep `dropna` only for target variables
- [ ] Code Reviewer: Confirm 5-fold CV loop in `main.py` runs end-to-end without errors on a small subset
- [ ] Code Reviewer: Add IBS and time-dependent AUC to results table (already in `metrics.py`, verify correct usage)
- [ ] Code Reviewer: Fix stroke endpoint — audit event rate (likely <3%), consider resampling or separate analysis
- [ ] Literature Scout: Survey survival DL papers 2018–2025 (DeepHit, DSM, SurvTRACE, DRSA, MENSA)
- [ ] Literature Scout: Survey tabular foundation models (TabPFN v1/v2, SAINT, TabICL, FT-Transformer)
- [ ] Supervisor: Draft core research question + paper narrative

**Week 2 (by 2026-04-06) — Baseline Benchmarking + Combo Grid Launch**
- [ ] Code Reviewer: Run full 5-fold CV on Sirbu for all current models (KM, Cox, DeepSurv, DeepHit, RSF, GBSA, TorchSurv)
- [ ] Code Reviewer: Record C-index, IBS, time-dep AUC for all models and all 4 endpoints
- [ ] Code Reviewer: Verify URRAH preprocessing pipeline matches Sirbu pipeline
- [ ] Code Reviewer: Run `experiments/tabpfn_combos.py` on GBSG, METABRIC, SUPPORT2 (external benchmarks) — confirm no crashes across all 6×6 combos
- [ ] Code Reviewer: Run `experiments/tabpfn_combos.py` on Sirbu (private) — record first combo-grid results; flag any OOM issues on URRAH
- [ ] Code Reviewer: Validate retrieval embedding variants (`tabpfn_retrieval_k10`, `tabpfn_retrieval_k50`) produce sensible C-index values on GBSG
- [ ] Literature Scout: Identify 3–5 closest related papers to frame our contribution; flag what they do NOT do
- [ ] Supervisor: Finalize novelty framing relative to SurvTRACE and TabPFN embedding papers
- [ ] Supervisor: Triage combo-grid results — identify top-performing (embedding, head) pairs for deeper ablation in Week 3

**Week 3 (by 2026-04-13) — Approach A Implementation**
- [ ] Code Reviewer: Extend `embedding_cox.py` pattern to DeepHit head (TabPFN embeddings → DeepHit)
- [ ] Code Reviewer: Implement `embedding_dsm.py` (TabPFN embeddings → DSM via auton-survival)
- [ ] Code Reviewer: Parameterize n_fold in {0, 3, 5} and log embedding dimensionality
- [ ] Code Reviewer: Run Approach A on Sirbu total-mortality task as a first sanity check
- [ ] Literature Scout: Compile calibration literature for survival models (reliability diagrams, D-calibration)

**Week 4 (by 2026-04-20) — URRAH Integration & Results Triage**
- [ ] Code Reviewer: Run full Approach A baseline sweep on both Sirbu and URRAH
- [ ] Code Reviewer: Compare embedding quality: n_fold=0 vs n_fold=5 on C-index
- [ ] Code Reviewer: Diagnose stroke endpoint — if event rate < 2%, flag as out of scope or use AUROC only
- [ ] Supervisor: Decide whether Approach B (fine-tuning) is feasible given Month 1 results
- [ ] Supervisor: Draft Related Work and Methods sections (skeleton)

---

### Month 2: Experiments & Analysis (2026-04-23 → 2026-05-22)

**Week 5–6 (by 2026-05-04) — Approach A Full Sweep + Ablations**
- [ ] Code Reviewer: Full 5-fold sweep on both datasets, all 4 endpoints, all survival heads (CoxPH, DeepHit, DSM)
- [ ] Code Reviewer: Ablation: TabPFN embedding vs. raw features (same model architecture)
- [ ] Code Reviewer: Ablation: frozen vs. adapter-fine-tuned embeddings (if feasible)
- [ ] Code Reviewer: Ablation: embedding dimensionality (n_fold 0/3/5/10)
- [ ] Code Reviewer: Calibration plots (D-calibration, reliability diagrams) for best models
- [ ] Literature Scout: Finalize related work citations; write annotated bibliography

**Week 7 (by 2026-05-11) — Approach B/C Exploration**
- [ ] Code Reviewer: Implement LoRA adapters on top of TabPFN (if Approach B)
- [ ] Code Reviewer: Implement ICL discretization survival (Approach C) as exploratory experiment
- [ ] Code Reviewer: Run Approach B/C on total-mortality task only (proof of concept)
- [ ] Supervisor: Decide which approach to lead with in the paper; write contribution bullet points

**Week 8 (by 2026-05-18) — Statistical Validation**
- [ ] Code Reviewer: Run Wilcoxon signed-rank tests across folds (already in `metrics.py`)
- [ ] Code Reviewer: Compute 95% CI for C-index using bootstrap (1000 samples)
- [ ] Code Reviewer: External consistency check: do URRAH results directionally match Sirbu?
- [ ] Literature Scout: Check CHIL / MLHC formatting requirements; confirm page limits and supplementary policies
- [ ] Supervisor: Write Introduction and paper abstract (draft)

**Week 9 (by 2026-05-22) — Results Consolidation**
- [ ] All agents: Results table finalized with mean ± std over 5 folds
- [ ] Code Reviewer: Generate all main figures (survival curves, calibration, embedding TSNE)
- [ ] Supervisor: Complete Methods section draft
- [ ] Supervisor: Complete Results section skeleton

---

### Month 3: Paper Writing & Submission (2026-05-23 → 2026-06-23)

**Week 10 (by 2026-05-29) — First Full Draft**
- [ ] Supervisor: Complete first full paper draft (all sections)
- [ ] Code Reviewer: Finalize reproducible code; ensure `main.py --tune` runs clean from scratch
- [ ] Code Reviewer: Write experiment README and config files

**Week 11 (by 2026-06-05) — Internal Review**
- [ ] All agents: Read and critique full draft
- [ ] Code Reviewer: Spot-check all numbers in tables against raw CSV results
- [ ] Supervisor: Revise based on feedback; tighten abstract and contributions

**Week 12 (by 2026-06-12) — Polish & Supplementary**
- [ ] Supervisor: Complete supplementary material (extended tables, additional ablations)
- [ ] Code Reviewer: Package code for anonymous submission (remove identifying paths/comments)
- [ ] Supervisor: Final proofread; format check against venue template

**Week 13 (by 2026-06-23) — SUBMISSION**
- [ ] Submit to CHIL 2026 (primary) or MLHC 2026 (secondary)
- [ ] Archive codebase at current state
- [ ] Tag git commit as `submission-v1`

---

## 4. Paper Structure Outline (8 pages + references)

**Title (working):** "Foundation Model Embeddings for Clinical Survival Analysis: A Competing-Risk Benchmark on Real-World Cardiovascular Cohorts"

### Abstract (~150 words)
- Problem: Survival analysis on clinical tabular data; few works use foundation model priors
- Method: TabPFN embeddings + survival heads (CoxPH, DeepHit, DSM) with proper competing-risk evaluation
- Datasets: Two independent cardiovascular cohorts (Sirbu n=8K, URRAH n=27K)
- Results: X% C-index improvement over classical Cox; IBS Y; statistically significant across 5-fold CV
- Conclusion: Foundation model embeddings provide a consistent, zero-configuration improvement for small clinical datasets

### 1. Introduction (~0.75 pages)
- Survival analysis in cardiology: why it matters, what is hard
- Classical methods (Cox PH, RSF) vs. deep survival models (DeepSurv, DeepHit)
- Gap: tabular foundation models (TabPFN) have not been evaluated for survival tasks
- Our contributions (bullet list):
  1. First systematic evaluation of TabPFN embeddings for clinical survival analysis
  2. Competing-risk benchmark across 4 endpoints on 2 independent cohorts
  3. Rigorous evaluation: 5-fold CV, C-index, IBS, time-dep AUC, calibration, statistical tests
  4. Public reproducible codebase

### 2. Related Work (~1 page)
- 2.1 Deep survival models: DeepSurv, DeepHit, DSM, DRSA, SurvTRACE
- 2.2 Tabular foundation models: TabPFN v1/v2, SAINT, FT-Transformer, TabICL
- 2.3 Foundation models in clinical/EHR settings (not survival-specific)
- 2.4 Gap statement: no prior work applies TabPFN-style embeddings to censored-time outcomes

### 3. Methods (~1.75 pages)
- 3.1 Problem formulation: competing risks, cause-specific hazard, CIF
- 3.2 Datasets: Sirbu (n=8,065, 4 outcomes) and URRAH (n=27,078), preprocessing, imputation strategy
- 3.3 TabPFN embedding extraction (frozen, K-fold variant to avoid contamination)
- 3.4 Survival heads: CoxPH (linear, proportional hazard), DeepHit (discrete-time, competing risks), DSM (mixture of Weibulls)
- 3.5 Baselines: Kaplan-Meier, Multivariate Cox, DeepSurv, RSF, GBSA, TorchSurv-Cox
- 3.6 Evaluation protocol: 5-fold stratified CV, Harrell's C-index, IBS, time-dep AUC, D-calibration, Wilcoxon signed-rank test

### 4. Results (~2 pages)
- 4.1 Main results table: C-index and IBS across all models × tasks × datasets (mean ± std)
- 4.2 TabPFN embedding ablation: n_fold comparison; embedding dim effect
- 4.3 Calibration results: reliability diagrams for best model per task
- 4.4 Statistical significance: Wilcoxon p-values vs. Multivariate Cox baseline
- 4.5 Stroke endpoint analysis: event rate discussion; note limitations

### 5. Discussion (~0.75 pages)
- When do foundation model embeddings help most? (sample size, event rate, feature complexity)
- Limitations: private datasets (no public replication), TabPFN context window, stroke endpoint
- Clinical implications: model deployment considerations, calibration importance
- Future work: end-to-end fine-tuning, MIMIC-III external validation, other foundation models

### 6. Conclusion (~0.25 pages)
- Summary of findings
- Reproducibility statement and code release

### References (~1 page)
- Target ~25–30 references

### Supplementary (unlimited)
- Full results tables for all folds
- Additional ablations (hyperparameter sensitivity)
- Preprocessing pipeline details
- Dataset statistics and missingness tables

---

## 5. Evaluation Protocol

### Datasets
| Dataset | N (raw) | N (clean) | Primary Endpoint | Split Strategy |
|---------|---------|-----------|-----------------|----------------|
| Sirbu | 8,065 | ~6,500 (after imputation) | Total mortality | 5-fold stratified CV |
| URRAH | 27,078 | ~24,000 (after imputation) | Total mortality | 5-fold stratified CV |

Stratification variable: joint (event indicator × time quartile bin).

### Models to Evaluate
**Classical baselines:**
- Kaplan-Meier (population-level, C-index = 0.5 by construction)
- Multivariate Cox PH (reference for all statistical tests)
- Random Survival Forest (RSF, 200 trees)
- Gradient Boosting Survival Analysis (GBSA)

**Deep baselines:**
- DeepSurv (MLP + Cox loss)
- TorchSurv-Cox (our custom implementation)
- DeepHit (discrete-time, competing risks)
- DSM (Deep Survival Machines)

**Proposed models (Approach A):**
- TabPFN-CoxPH (frozen embedding + CoxPH head)
- TabPFN-DeepHit (frozen embedding + DeepHit head)
- TabPFN-DSM (frozen embedding + DSM head)
- TabPFN-CoxPH-KFold (K-fold embedding + CoxPH head, n_fold=5)

### Metrics
| Metric | Description | Tool |
|--------|-------------|------|
| C-index (Harrell) | Discrimination | `sksurv.metrics.concordance_index_censored` |
| C-index (Antolini) | Time-dependent discrimination | `pycox.evaluation.EvalSurv` |
| IBS | Integrated Brier Score (calibration + discrimination) | `sksurv.metrics.integrated_brier_score` |
| Time-dep AUC | AUC at 25th/50th/75th event percentile | `sksurv.metrics.cumulative_dynamic_auc` |
| D-calibration | Calibration of discrete survival bins | Manual implementation |

All metrics reported as mean ± std over 5 folds. Bootstrap CI (1000 samples) for test-set metrics.

### Statistical Tests
- **Wilcoxon signed-rank test** (already implemented in `metrics.py`): Compare each model vs. Multivariate Cox across 5 folds
- Significance threshold: p < 0.05 (two-sided); Bonferroni correction for multiple endpoints
- **Paired t-test** as sensitivity analysis (normality not guaranteed → Wilcoxon preferred)

### Preprocessing Protocol (Revised)
1. Exclude outcome-collinear columns before any modeling (audit against CLAUDE.md bug list item #1)
2. Impute continuous features with `IterativeImputer` (max_iter=10, estimator=ExtraTreesRegressor) using train-fold statistics only
3. Binary features: mode imputation on train fold, apply to test fold
4. Standardize continuous features: StandardScaler fit on train fold only
5. Stratified split for CV: strata = `event_indicator + pd.qcut(duration, 4)`

---

## 6. Risk Assessment

### Risk 1: TabPFN context window / sample-size limitation
**Probability: Medium | Impact: High**

TabPFN v2 supports up to ~10,000 training samples and ~500 features per inference call. The URRAH dataset (27K samples) exceeds this.

Mitigation:
- For URRAH, sub-sample training context to 8,000 rows per fold (reproducible, seeded)
- Report sensitivity: vary context size from 2K → 8K → full (sub-sampled)
- Alternatively, use TabPFN in bag-of-contexts mode (multiple random subsets, average embeddings)
- If TabPFN fails on URRAH, Approach A is restricted to Sirbu only; frame URRAH as a held-out generalization test using a simpler model

### Risk 2: Stroke endpoint is broken (C-index ≈ 0.38)
**Probability: High | Impact: Medium**

Stroke events are likely very rare (<2% of Sirbu). At low event rates, any model collapses toward random.

Mitigation:
- Compute exact event rate; if <3%, exclude from primary results table with explicit justification
- Report stroke as supplementary with AUROC as the primary metric (more stable at low event rates)
- Consider case-control subsampling for stroke to balance event rates during training

### Risk 3: No significant improvement from TabPFN embeddings
**Probability: Medium | Impact: High**

If C-index improvement is <0.01 and not statistically significant, the primary claim collapses.

Mitigation:
- Ensure evaluation protocol is genuinely rigorous (proper imputation, CV, IBS) — even a negative result is publishable at MLHC if the benchmark is valuable
- Pivot narrative: "We conduct the first rigorous benchmark of foundation model embeddings for survival analysis and find X" — a null result with proper methodology is still a contribution
- Explore where TabPFN helps most: small datasets, high missingness, rare events → turn into a subgroup analysis
- Pursue Approach B (fine-tuning) as a stronger intervention if Approach A shows modest gains

### Risk 4: Dataset access / privacy constraints block reproducibility
**Probability: Low–Medium | Impact: Medium**

Sirbu and URRAH are private datasets. Reviewers cannot verify results independently.

Mitigation:
- CHIL and MLHC both accept private-dataset papers with institutional data use agreements cited
- Provide full reproducible code for the method (not data)
- Run an additional experiment on a public dataset (GBSG, METABRIC, or SEER) as a reproducibility anchor
- Include detailed dataset statistics (Table 1) so results are contextually interpretable

### Risk 5: Timeline slippage from infrastructure bugs
**Probability: High | Impact: Medium**

The codebase has 9 documented bugs (see CLAUDE.md). Each may compound.

Mitigation:
- Week 1 is dedicated entirely to bug fixes — no new experiments until bugs are resolved
- Implement automated test suite: unit test for each model (`test_tabpfn.py` already exists as a template)
- Use `main.py --folds 2 --trials 2` as a smoke-test that runs in <5 minutes
- Set a hard cutoff: if a bug takes >2 days to fix, deprioritize that model/endpoint

### Risk 6: MIMIC-III integration remains incomplete
**Probability: High | Impact: Low**

External validation on MIMIC-III was planned but scaffolding is unfinished.

Mitigation:
- Deprioritize MIMIC-III entirely; not needed for a strong CHIL/MLHC submission
- Frame URRAH as the external validation cohort (different hospital/country from Sirbu)
- If time permits in Month 3, implement MIMIC-III as a supplementary analysis only

---

## 7. Target Venues

### Primary: CHIL 2026
**Full name:** ACM Conference on Health, Inference, and Learning
**Submission deadline:** ~2026-06-23 (estimated; confirm at chil.acm.org)
**Notification:** ~2026-08-15 (estimated)
**Format:** 8 pages + unlimited references; SIGCONF format
**Fit:** Directly targets clinical ML with rigorous evaluation; clinical datasets welcome; no public data required
**Strength of fit:** Very High

### Secondary: MLHC 2026
**Full name:** Machine Learning for Healthcare
**Submission deadline:** ~2026-05-01 (estimated; confirm at mlforhc.org — NOTE: may be earlier than CHIL)
**Notification:** ~2026-07-01 (estimated)
**Format:** 10 pages; PMLR format
**Fit:** Overlapping scope; stronger focus on clinical validation; private data is standard
**Strength of fit:** High
**Warning:** MLHC deadline may fall in late April/early May — verify immediately and adjust Month 2 plans if needed.

### Fallback: NeurIPS 2026 Workshop
**Relevant workshops:** Health in the Age of LLMs; Tabular Learning; Clinical NLP
**Submission deadline:** ~2026-09 (workshop-dependent)
**Format:** 4–8 pages
**Fit:** Lower bar; good for preliminary results if CHIL/MLHC rejection; keeps the work moving

### Venue Comparison Table
| Venue | Deadline | Pages | Review Style | Private Data | Recommendation |
|-------|----------|-------|-------------|-------------|----------------|
| CHIL 2026 | ~2026-06-23 | 8 | Double-blind | Accepted | PRIMARY |
| MLHC 2026 | ~2026-05-01 | 10 | Double-blind | Standard | SECONDARY (verify deadline!) |
| NeurIPS Workshop | ~2026-09 | 4–8 | Single-blind | Accepted | FALLBACK |

**Action required:** Verify exact deadlines for CHIL 2026 and MLHC 2026 by 2026-03-30. If MLHC deadline is in May, restructure Month 1–2 to target MLHC as primary.

---

## Appendix: Key Architectural Decisions Already Made

Based on the codebase audit:

1. `main.py` implements 5-fold stratified CV — this is already in place (good)
2. `metrics.py` implements C-index, IBS, time-dep AUC — already in place (good)
3. `embedding_cox.py` implements TabPFN → CoxPH — partially working; needs DeepHit and DSM extensions
4. `preprocessing.py` uses `dropna()` as the final step — must be replaced with imputation
5. `preprocessing.py` applies `StandardScaler` inside task preparation functions — risk of data leakage if scaler is fit on full dataset before split; must be moved into the fold loop in `main.py`
6. `tabpfn/embedding.py` provides `TabPFNEmbedding` with K-fold extraction — ready to use
7. `survpfn/tabpfn_modeling/` contains custom transformer/encoder code — relationship to main pipeline unclear; requires Code Reviewer audit
8. Optuna hyperparameter tuning is implemented for each model — use `--tune --trials 20` for final experiments

---

*This document is the authoritative project plan. All agents should update the `## Agent Progress` section in CLAUDE.md as milestones are completed.*
