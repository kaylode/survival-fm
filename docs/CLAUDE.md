# SurvPFN Research Project — CLAUDE.md

> Living document. Updated by agents as they make progress.
> Goal: Publish at a top health ML conference (e.g., CHIL, MLHC, NeurIPS Health) within **3 months** (deadline ~2026-06-23).

---

## Project Overview

Survival analysis on two private clinical datasets using foundation model embeddings + survival heads. Key hypothesis: foundation models (e.g., TabPFN, TabICL, SAINT, etc.) can extract richer tabular embeddings than classic feature engineering, improving survival predictions.

---

## Datasets

### Sirbu Dataset (Primary)
- **Size:** 8,065 samples → 6,264 after `dropna()` cleaning
- **Features:** 85 columns, 78 used for modeling
- **Outcomes:** Total mortality, CVD mortality, MI, Stroke (competing risks)
- **Split:** 80/10/10 train/val/test (stratified by event + time quartile)
- **Issues:** Up to 99% missing in LVH, VES; 30% rows dropped by aggressive `dropna()`

### URRAH Dataset (Secondary)
- **Size:** 27,078 samples, 85 columns
- **Features:** Similar clinical variables to Sirbu
- **Status:** Loaded, inconsistencies identified, not yet used for modeling
- **Issues:** Same high-missingness features (LVH, VES)

### MIMIC-III (External Validation — Incomplete)
- Code scaffolded but not integrated
- Intended for external validation

---

## Current Baselines & Results

| Model | Outcome | C-index |
|-------|---------|---------|
| Multivariate Cox PH | Total mortality | ~0.75 (est.) |
| DeepSurv | Total mortality | **0.7793** |
| DeepHit | CVD mortality | 0.7583 |
| DeepHit | MI | 0.6420 |
| DeepHit | Stroke | 0.3831 ⚠️ |

---

## Critical Bugs & Issues (Code Reviewer Priority)

1. **Data leakage risk** — outcome-related columns not cleanly excluded
2. **Aggressive dropna** — loses 22% of data; imputation needed
3. **No cross-validation** — single split, no confidence intervals
4. **No hyperparameter tuning** — fixed defaults throughout
5. **Missing evaluation metrics** — only C-index; need IBS, calibration, Brier score
6. **Stroke model broken** — C-index 0.38, likely data quality or low event rate
7. **URRAH not used for modeling** — only loaded/inspected
8. **No requirements.txt / pyproject.toml** — environment not reproducible
9. **All code in one notebook** — no modular scripts

---

## Finalized Research Question

**Can tabular foundation model embeddings (TabPFN, SAINT) significantly improve survival analysis performance on small-to-medium clinical datasets, relative to classical feature inputs, when paired with purpose-built survival heads under a rigorous competing-risk evaluation protocol?**

We test whether frozen or lightly fine-tuned in-context representations from TabPFN v2 provide a strictly superior inductive bias for survival tasks compared to raw features fed into the same survival architectures (DeepSurv, DeepHit, DSM), and whether the gain is consistent across two independent cohorts (Sirbu n=8,065; URRAH n=27,078) and four clinical endpoints (total mortality, CVD mortality, MI, stroke).

---

## Candidate Approaches (Ranked by Novelty × Feasibility)

### Approach A — TabPFN Embedding + Survival Head (PRIMARY)
**Status: Implementation started (`embedding_cox.py`). DeepHit and DSM heads pending.**

```
Raw clinical features
    → TabPFN v2 (frozen, K-fold embedding extraction, n_fold in {0,3,5})
    → Embedding vector (~512-dim)
    → Survival Head (CoxPH / DeepHit / DSM)
    → S(t|x) or CIF(t|x)
```

Key choices: Use `TabPFNClassifier` (event vs. censored) as embedding proxy task. Compare n_fold variants to quantify embedding quality.

### Approach B — End-to-End Fine-Tuning with Survival Loss (SECONDARY)
**Status: Not started. Feasibility depends on Month 1 results.**

Fine-tune last N transformer layers of TabPFN using differentiable survival loss (DeepHit / DSM). High-risk due to TabPFN's in-context learning architecture. LoRA adapters are the pragmatic fallback.

### Approach C — TabPFN ICL via Time Discretization (EXPLORATORY)
**Status: Concept only. Allocate 2 weeks in Month 2 if Approach A is strong.**

Discretize time → K-class classification → TabPFN in-context inference → discrete survival function. No training required; pure ICL. Limited by context window (~10K rows).

---

## Literature Gaps to Fill (Scout Priority)

- [ ] Foundation models for tabular survival analysis (very sparse)
- [ ] TabPFN for survival / censored data
- [ ] Deep learning competing risks (DeepHit, DSM, DRSA)
- [ ] Self-supervised pre-training on clinical tabular data
- [ ] Calibration in survival models
- [ ] Benchmarks for EHR survival prediction

---

---

## Agent Progress

> Updated by agents as work is completed. Format: `[YYYY-MM-DD] AgentName: what was done.`

### Supervisor
- [2026-03-23] Supervisor: Audited full codebase (`main.py`, `embedding_cox.py`, `metrics.py`, `preprocessing.py`, `deep_hit.py`, `custom_model.py`, `tabpfn/embedding.py`). Confirmed 5-fold CV loop is in place, IBS/AUC metrics are implemented, TabPFN embedding extractor is functional for CoxPH. Identified critical data-leakage risk in `preprocessing.py` (StandardScaler fit before fold split).
- [2026-03-23] Supervisor: Created `RESEARCH_PLAN.md` with full 3-month timeline, paper outline, evaluation protocol, risk assessment, and venue targets.
- [2026-03-23] Supervisor: Finalized research question and three candidate approaches; updated CLAUDE.md.
- [2026-03-23] Supervisor: Designed 6×6 TabPFN combo experiment matrix, created `experiments/tabpfn_combos.py` (6 embedding variants × 6 survival heads, 5-fold CV, external benchmarks GBSG/METABRIC/SUPPORT2 + private Sirbu/URRAH), updated `RESEARCH_PLAN.md` with experimental matrix (§2b), Approach D retrieval-augmented TabPFN (§2 Approach D), deferred meta-training note (§2c), and updated Week 2 milestones to include benchmark datasets running and combo grid launch.

### Code Reviewer
- [2026-03-23] Code Reviewer: Read all files in `survpfn/` + `main.py` + `pyproject.toml`. Fixed BUG-1 (CRITICAL): removed `StandardScaler` from all four `prepare_*` functions in `preprocessing.py` — scaler was fit on the full dataset before the CV fold split, leaking test statistics into training. Fixed BUG-2 (CRITICAL): extended drop lists in all four `prepare_*` functions to remove post-baseline event flag columns (`CABG _event`, `Non Fatal AMI (Follow-Up)_event`, `Ictus_event`, `PCI_event`) and their date columns, which were outcome-correlated features retained in the feature matrix. Wrote `CODE_REVIEW.md` with full architecture diagram, 10 bugs catalogued with file:line references, assessments of `tabpfn/embedding.py` (correct for n_fold=0 vanilla; K-fold out-of-fold variant not used), CV setup (structurally correct after BUG-1 fix), metrics (IBS+AUC present; calibration missing), and `pyproject.toml` (missing `matplotlib`, `scipy`, `requests`, `cycler`).

### Literature Scout
- [2026-03-23] Scout: Created `LITERATURE_SURVEY.md` — full survey of 15+ papers across classical methods, DL survival (DeepSurv/DeepHit/DSM/SurvTRACE), tabular FMs (TabPFN/SAINT/TabNet). Confirmed SurvPFN gap is novel.
- [2026-03-23] Scout: Appended §8–10 to `LITERATURE_SURVEY.md`. **Retrieval-augmented SurvPFN is novel** — TabR (tabular retrieval), k-NN KM (survival retrieval), and KATE (ICL retrieval) each exist but are never combined; no prior work does retrieval-augmented ICL for censored survival. **Recommendation: publish SurvPFN + retrieval as one joint contribution** (retrieval solves TabPFN's N>1000 scalability limit). **Meta-training estimate: 31–49 person-days, 4–6 weeks with 2 engineers, ~$300–1500 GPU cost** — feasible but risky; highest risk is prior calibration. Marked as deferred to future work unless team grows.

---

## Team Roles

| Agent | Role |
|-------|------|
| **Supervisor** | Orchestrate agents, set milestones, resolve conflicts, write paper outline |
| **Code Reviewer** | Audit codebase, fix bugs, refactor, run new experiments |
| **Literature Scout** | Deep literature dive, identify gaps, surface novel directions |

---

## Progress Tracker

### Week 1 (by 2026-03-30)
- [ ] Literature scout: Complete survey of survival DL papers (2018–2025)
- [ ] Code reviewer: Fix top 5 bugs, add cross-validation, add IBS metric
- [ ] Supervisor: Draft research questions and paper outline

### Week 2–4 (by 2026-04-20)
- [ ] Implement foundation model embedding pipeline
- [ ] Run baseline experiments with proper evaluation
- [ ] Draft related work section

### Month 2 (by 2026-05-23)
- [ ] Novel method experiments (TabPFN + survival head)
- [ ] Ablation studies
- [ ] External validation on URRAH / MIMIC

### Month 3 (by 2026-06-15)
- [ ] Paper writing
- [ ] Final experiments
- [ ] Submission

---

## Target Venues

1. **CHIL 2026** — ACM Conference on Health, Inference, and Learning
2. **MLHC 2026** — Machine Learning for Healthcare
3. **NeurIPS 2026 (workshops)** — Health AI / EHR workshops
4. **AAAI 2026** — AI for health track

---

## Key References to Read

- DeepHit (Lee et al., 2018)
- DeepSurv (Katzman et al., 2018)
- DSM — Deep Survival Machines (Nagpal et al., 2021)
- DRSA (2022)
- TabPFN (Hollmann et al., 2023)
- SAINT (Somepalli et al., 2021)
- SurvTRACE (Wang et al., 2022)
- TabICL (2024)
- SurvivalGAN / survival augmentation papers

---

## Environment

- Use `uv` for all Python package management (see SKILL.md)
- All experiments should be reproducible with a `pyproject.toml`
