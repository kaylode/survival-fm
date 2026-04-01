# Literature Survey: Survival Analysis with ML/DL/Foundation Models

> **Scope:** Classical methods through foundation models, targeting CHIL 2026 / MLHC 2026.
> **Knowledge cutoff:** August 2025. Claims marked `[verify]` should be independently checked before submission.

---

## 1. Classical Survival Analysis Methods

### 1.1 Kaplan-Meier Estimator
- **Key paper:** Kaplan, E.L. & Meier, P. (1958). "Nonparametric estimation from incomplete observations." *Journal of the American Statistical Association*, 53(282), 457–481.
- **Main idea:** Non-parametric estimator of the survival function S(t) = P(T > t). Constructs a step function from ordered event times, accounting for censored observations at each step. No covariates — purely population-level.
- **Limitations:** Cannot incorporate covariates; treats all patients identically; produces only a group-level curve; no predictive C-index in the covariate sense.
- **Typical C-index:** 0.50 (by definition — no risk stratification without covariates). When used as stratified KM with binary covariates, C-index is a function of the covariate's predictive power.

### 1.2 Cox Proportional Hazards (Cox PH)
- **Key paper:** Cox, D.R. (1972). "Regression models and life-tables." *Journal of the Royal Statistical Society: Series B*, 34(2), 187–220.
- **Main idea:** Semi-parametric model: h(t|x) = h_0(t) · exp(β^T x). The baseline hazard h_0(t) is left unspecified; parameters β are estimated via partial likelihood, avoiding explicit modeling of time. Linearity assumption in log-hazard space. Proportional hazards assumption: covariate effects are constant over time.
- **Limitations:** Proportional hazards assumption is frequently violated in practice; linear log-hazard cannot capture complex interactions; requires complete covariate vectors; struggles with high-dimensional data without regularization.
- **Typical C-index:** Dataset-dependent; on SUPPORT2: ~0.60–0.63 [verify]; on METABRIC: ~0.63–0.66 [verify]; on GBSG: ~0.66–0.68 [verify].

### 1.3 Lasso-Cox (Regularized Cox)
- **Key paper:** Tibshirani, R. (1997). "The lasso method for variable selection in the Cox model." *Statistics in Medicine*, 16(4), 385–395.
- **Main idea:** Adds an L1 penalty λ‖β‖_1 to the Cox partial log-likelihood. Simultaneously performs variable selection and shrinkage, enabling high-dimensional survival regression (e.g., genomic data with p >> n).
- **Variations:** Ridge-Cox (L2), ElasticNet-Cox (L1+L2). The `glmnet` R package (Simon et al., 2011) is the standard implementation.
- **Limitations:** Still assumes log-linear hazard; variable selection can be unstable across folds; does not model non-linear or interaction effects.
- **Typical C-index:** Marginally better than Cox PH on high-dimensional datasets; on clinical tabular data, gains are modest (~0.01–0.02) [verify].

### 1.4 Random Survival Forests (RSF)
- **Key paper:** Ishwaran, H., Kogalur, U.B., Blackstone, E.H., & Lauer, M.S. (2008). "Random survival forests." *Annals of Applied Statistics*, 2(3), 841–860.
- **Main idea:** Ensemble of survival trees, each trained on a bootstrap sample using a random feature subset. Splitting criterion: log-rank test statistic. Terminal nodes store the Nelson-Aalen cumulative hazard estimate. The ensemble averages individual tree cumulative hazard functions.
- **Implementation:** `randomForestSRC` R package; `scikit-survival` Python port.
- **Limitations:** Computationally expensive at scale; less interpretable than Cox; still assumes feature interactions are tree-representable; does not produce a smooth survival function.
- **Typical C-index:** Often the strongest classical baseline: SUPPORT2 ~0.67–0.72 [verify]; METABRIC ~0.66–0.68 [verify]. Frequently competitive with or superior to early deep learning approaches.

### 1.5 Gradient Boosting for Survival
- **Key papers:**
  - **GBM-Cox:** Ridgeway, G. (1999). "The state of boosting." *Computing Science and Statistics*, 31, 172–181. Cox loss in gradient boosting machines.
  - **CoxBoost:** Binder & Schumacher (2008). "Allowing for mandatory covariates in boosting estimation of sparse high-dimensional survival models." *BMC Bioinformatics*.
  - **XGBoost-AFT/Cox:** Chen & Guestrin (2016) XGBoost framework extended with AFT (accelerated failure time) and Cox objectives. `xgboost` supports `survival:cox` and `survival:aft` objectives.
  - **LightGBM-Cox:** Analogous survival objectives in LightGBM.
- **Main idea:** Boosted trees minimize a survival loss function (partial log-likelihood for Cox, or AFT log-likelihood) via functional gradient descent. Captures non-linear effects and interactions without manual feature engineering.
- **Limitations:** XGBoost-Cox uses Breslow approximation for ties; no native competing-risks objective; calibration requires post-hoc adjustment; hyperparameter sensitivity.
- **Typical C-index:** Competitive with RSF; often 0.01–0.03 above RSF on tabular data [verify].

---

## 2. Deep Learning for Survival Analysis

### 2.1 Single-Risk Models

#### DeepSurv
- **Key paper:** Katzman, J.L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). "DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network." *BMC Medical Research Methodology*, 18(1), 1–12.
- **Architecture:** Fully connected feedforward network (MLP) with dropout and batch normalization. Output: a single scalar log-risk score h = f_θ(x). No time input; mimics linear Cox but with non-linear feature mapping.
- **Loss:** Negative Cox partial log-likelihood (Breslow approximation for tied event times): L = -Σ_i [ h_i - log Σ_{j∈R(t_i)} exp(h_j) ] where R(t_i) is the risk set at time t_i.
- **Datasets used:** WHAS, GBSG (German breast cancer), METABRIC, Rotterdam, SUPPORT2.
- **Reported C-index:** SUPPORT2 ~0.6174 [verify]; METABRIC ~0.6518 [verify]; GBSG ~0.6591 [verify]. Often cited as marginal improvement over linear Cox.
- **Limitations:** Inherits PH assumption from Cox loss; cannot extrapolate beyond observed time range; batch construction for partial likelihood is O(n²) in naive implementations; no competing risks.

#### MTLR / N-MTLR
- **Key papers:**
  - Yu, C.N., Greiner, R., Lin, H.C., & Baracos, V. (2011). "Learning patient-specific cancer survival distributions as a sequence of dependent regressors." *NeurIPS*, 24.
  - Fotso, S. (2018). "Deep neural networks for survival analysis based on a multi-task framework." arXiv:1801.05512. (Neural extension: N-MTLR)
- **Main idea:** Discretizes time into K intervals. Fits K logistic regression models (one per interval) sharing parameters, with a constraint ensuring monotonicity of the survival function (adjacent logistic outputs are dependent). N-MTLR replaces the linear model with a neural network backbone.
- **Formulation:** P(T ∈ [t_{k-1}, t_k) | x) is modeled as a softmax over K+1 classes (survival beyond all intervals + each interval). The survival function S(t_k | x) = Σ_{j>k} P(T ∈ interval_j | x).
- **Reported C-index:** Competitive with DeepSurv on SUPPORT2 and METABRIC [verify].
- **Limitations:** Discretization granularity is a hyperparameter; K large → computation grows; does not naturally extend to competing risks without modification.

#### PC-Hazard
- **Key paper:** Kvamme, H. & Borgan, Ø. (2019). "Continuous and discrete-time survival prediction with neural networks." arXiv:1910.06724. Published in *Lifetime Data Analysis*, 2021.
- **Main idea:** Piecewise-constant hazard model. Discretizes time into intervals; within each interval hazard is constant. Uses a neural network to predict interval-specific hazards. Bridges the gap between discrete and continuous time models. Loss is the exact discrete-time likelihood under piecewise constant hazard assumption.
- **Advantage over DeepSurv:** Does not require the PH assumption; hazard can vary freely across time intervals per individual.
- **Reported C-index:** On METABRIC and SUPPORT2, typically competitive with or slightly above DeepSurv [verify].
- **Limitations:** Interval granularity must be tuned; too coarse → loss of temporal resolution; too fine → sparse events per interval.

#### Cox-Time
- **Key paper:** Kvamme, H., Borgan, Ø., & Scheel, I. (2019). "Time-to-event prediction with neural networks and Cox regression." *Journal of Machine Learning Research*, 20(129), 1–30.
- **Main idea:** Extends the Cox model to allow time-varying relative risk: h(t|x) = h_0(t) · exp(f_θ(x, t)). The neural network receives both the covariate vector x and the evaluation time t as input, relaxing the proportional hazards assumption. Uses a sampled softmax approximation to the partial likelihood for computational efficiency.
- **Key contribution:** First Cox-based deep model that relaxes PH by conditioning on time explicitly.
- **Reported C-index:** On METABRIC ~0.63–0.64 [verify]; on SUPPORT2 ~0.62–0.64 [verify].
- **Limitations:** Training requires sampling negative time points; still uses Cox-inspired loss (partial likelihood); less interpretable than standard Cox.

---

### 2.2 Competing Risks Models

#### DeepHit
- **Key paper:** Lee, C., Zame, W.R., Yoon, J., & van der Schaar, M. (2018). "DeepHit: A deep learning approach to survival analysis with competing risks." *AAAI Conference on Artificial Intelligence*, 32(1).
- **Architecture:** Multi-layer neural network with shared sub-network + cause-specific sub-networks. Output: a joint distribution over (cause, time) on a discrete time grid — specifically, a matrix P[cause k, time bin t] = probability of event type k in bin t.
- **Loss:** Two-component loss:
  1. **Likelihood loss:** Negative log-likelihood of the joint (cause, time) distribution. For censored subjects, uses the complementary probability (probability of surviving past censoring time across all causes).
  2. **Ranking loss:** Cause-specific concordance-based ranking loss to ensure predicted risks are correctly ordered across subjects.
  - L_total = L_likelihood + α · L_ranking
- **Datasets:** METABRIC (breast cancer survival, two competing risks), SEER (cancer registry).
- **Reported C-index:** METABRIC cause 1: ~0.69–0.71 [verify]; SEER: varies by cause [verify]. Widely cited as a strong competing-risks baseline.
- **Key contribution:** First major deep learning model to jointly model all competing risks in a single network with a combined loss.
- **Limitations:** Discrete time grid requires careful binning; ranking loss adds a hyperparameter α; can be miscalibrated (see Section 2.3); memory scales with K × T where K = causes and T = time bins.

#### DSM / Deep Survival Machines
- **Key paper:** Nagpal, C., Li, X., & Dubrawski, A. (2021). "Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks." *Journal of Machine Learning Research*, 22(1), 1–56.
- **Main idea:** Models the survival time distribution as a mixture of K Weibull (or log-normal) distributions: S(t|x) = Σ_k π_k(x) · S_Weibull(t; α_k, β_k). A neural network learns mixing weights π_k(x) and shape/scale parameters per component per covariate vector. Competing risks variant: each cause gets its own mixture.
- **Loss:** Evidence lower bound (ELBO)-style objective combining:
  - Log-likelihood of observed events under the mixture.
  - Auxiliary discriminative loss to enforce clustering in latent space (matched to the mixture components).
- **Key contributions:** Fully parametric (can extrapolate beyond observed times); interpretable mixture components; competing risks naturally handled.
- **Reported C-index:** On SUPPORT2 ~0.67–0.70 [verify]; on METABRIC ~0.64–0.67 [verify]. Competitive with DeepHit.
- **Limitations:** Number of mixture components K is a hyperparameter; Weibull assumption within components may not hold; ELBO objective can be harder to optimize than direct likelihood.

#### DRSA (Dynamic Recalibration Survival Analysis)
- **Key paper:** [verify — exact paper citation uncertain; the acronym DRSA appears in several survival analysis contexts. Possible reference: Chapfuwa et al. work on calibration, or Goldstein et al. dynamic survival analysis. Mark for verification.]
- **Main idea:** Addresses the observation that deep survival models are often miscalibrated (predicted survival probabilities don't match empirical frequencies). Proposes a recalibration procedure — typically a post-hoc isotonic regression or temperature scaling applied to predicted survival probabilities at each time point.
- **Note:** This area intersects heavily with calibration literature (see Section 2.3). Multiple groups have proposed "dynamic" recalibration schemes for survival models where the recalibration function is itself time-dependent.

#### SurvTRACE
- **Key paper:** Wang, Z., Sun, J., & Zhan, A. (2022). "SurvTRACE: Transformers for survival analysis with competing events." *ACM Conference on Health, Inference, and Learning (CHIL)*, 2022.
- **Architecture:** Transformer encoder (BERT-style) operating on tabular features. Each feature is tokenized (feature index + value embedding). Multi-head self-attention over feature tokens captures pairwise feature interactions. Output CLS token is passed to cause-specific prediction heads for discrete-time survival.
- **Pretraining:** BERT-style masked feature prediction (masking out a fraction of feature values and predicting them), providing self-supervised pretraining on the survival table itself.
- **Loss:** Combining: (1) Survival likelihood loss (similar to DeepHit-style discrete-time negative log-likelihood), (2) Masked feature prediction loss during pretraining.
- **Datasets:** METABRIC (two-event competing risks), SEER (multi-cause mortality).
- **Reported C-index:** METABRIC: [verify: ~0.69–0.72]; SEER: [verify]. Reports improvement over DeepHit, DSM, and RSF baselines.
- **Key contributions:** First transformer architecture applied to tabular survival data with competing risks; demonstrates that self-supervised pretraining on survival tables improves downstream predictions.
- **Limitations:** In-domain pretraining only (not a general foundation model); still requires the same training distribution; tokenization of continuous features is non-trivial; performance gains over DeepHit are modest on some datasets [verify].

---

### 2.3 Calibration & Evaluation Metrics

#### Concordance Index (C-index / Harrell's C)
- **Reference:** Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., & Rosati, R.A. (1982). "Evaluating the yield of medical tests." *JAMA*, 247(18), 2543–2546.
- **Definition:** Proportion of all comparable pairs where the model correctly orders risk: C = P(risk_i > risk_j | T_i < T_j, not censored). Ranges [0.5, 1.0] for useful models; 0.5 = random.
- **Limitations:** Does not evaluate calibration; insensitive to absolute risk levels; affected by censoring rate; only measures discrimination.
- **Competing risks variant:** Cause-specific C-index and cause-specific time-dependent AUC are more appropriate.

#### Integrated Brier Score (IBS)
- **Reference:** Graf, E., Schmoor, C., Sauerbrei, W., & Schumacher, M. (1999). "Assessment and comparison of prognostic classification schemes for survival data." *Statistics in Medicine*, 18(17-18), 2529–2545.
- **Definition:** IBS = (1/t_max) ∫_0^{t_max} BS(t) dt where BS(t) = (1/n) Σ_i w_i · [I(T_i ≤ t, δ_i=1) - Ŝ(t|x_i)]². Inverse probability of censoring weighting (IPCW) handles censored observations.
- **Advantage over C-index:** Measures calibration + discrimination jointly; penalizes models that predict wrong probability levels.
- **Typical range:** Well-calibrated models: IBS < 0.25; near-random: IBS ≈ 0.25 [verify for specific datasets].

#### D-Statistic
- **Reference:** Royston, P. & Sauerbrei, W. (2004). "A new measure of prognostic separation in survival data." *Statistics in Medicine*, 23(5), 723–748.
- **Definition:** Measures separation between high- and low-risk groups defined by the prognostic index. Related to the log hazard ratio between the upper and lower half of the predicted risk score distribution. Reported in log-hazard units.

#### Time-Dependent AUC
- **References:** Heagerty, P.J. & Zheng, Y. (2005). "Survival model predictive accuracy and ROC curves." *Biometrics*, 61(1), 92–105. Uno et al. (2007) provide a censoring-robust version.
- **Definition:** At each time point t, AUC(t) = P(risk_i > risk_j | T_i ≤ t, T_j > t). Integrated time-dependent AUC (iAUC) summarizes across time.
- **Advantage:** Time-specific discrimination; reveals model performance degradation over time.

#### Calibration Plots for Survival
- **Reference:** van Calster, B. et al. (2016) and Austin et al. (2020) on survival calibration.
- **Methods:** D-calibration (Haider et al., 2020, *JMLR*): tests whether the distribution of predicted survival probabilities is uniform. Graphical calibration: plot Kaplan-Meier curves for deciles of predicted risk vs. predicted mean risk per decile. Expected Calibration Error (ECE) adapted for survival.
- **State of the field:** Deep survival models are frequently found to be poorly calibrated despite high C-index [verify: Haider et al. 2020 showed this systematically]. Calibration is a significant open problem.

---

## 3. Foundation Models & Self-Supervised Learning for Tabular Data

### 3.1 Key Tabular Foundation Models

#### TabNet
- **Key paper:** Arik, S.Ö. & Pfister, T. (2021). "TabNet: Attentive interpretable tabular learning." *AAAI Conference on Artificial Intelligence*, 35(8), 6679–6687.
- **Architecture:** Sequential multi-step attention mechanism. At each step, a sparsemax-based attention mask selects a subset of features to process. Features pass through a gated linear unit (GLU). Multiple steps accumulate feature representations.
- **Self-supervised pretraining:** TabNet can be pretrained with a masked feature reconstruction objective on unlabeled tabular data (TabNet encoder-decoder).
- **Claimed strengths:** Instance-wise feature selection (interpretable masks), handles mixed feature types, end-to-end differentiable.
- **Limitations:** Sequential steps create optimization difficulties; hyperparameter sensitive (number of steps, sparsity regularization); often does not outperform well-tuned XGBoost on standard benchmarks; the interpretability claim is debated.
- **Survival application:** Not designed for survival; could be adapted by replacing the output layer with a Cox/DeepHit head, but no standard implementation exists [verify].

#### SAINT (Self-Attention and Intersample Attention Transformer)
- **Key paper:** Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C.B., & Goldstein, T. (2021). "SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training." arXiv:2106.01342.
- **Architecture:** Two types of attention applied sequentially:
  1. **Column attention (self-attention):** Standard transformer attention over feature tokens within a row — captures feature interactions.
  2. **Row attention (intersample attention):** Attention across different samples (rows) in a mini-batch — allows the model to compare a new row against training examples (conceptually similar to k-NN in embedding space).
- **Pretraining:** Contrastive learning with augmented views of tabular rows (CutMix-style corruption of feature values), plus masked feature prediction.
- **Reported performance:** Competitive with or superior to XGBoost on several UCI benchmarks [verify: specific margins uncertain].
- **Limitations:** Intersample attention is O(n²) in batch size; computationally expensive; not designed for survival data; no public survival-specific extensions found as of August 2025 [verify].

#### TabPFN
- **Key paper:** Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). "TabPFN: A transformer that solves small tabular classification problems in a second." *International Conference on Learning Representations (ICLR)*, 2023. Also appeared in *Nature* [verify: the Nature publication may be a subsequent extended version — verify exact venue].
- **Architecture:** Transformer trained via meta-learning on synthetic classification tasks generated from a Bayesian prior over data-generating processes. At inference time, the entire training set is provided as context (in-context learning): the model performs a forward pass that conditions predictions for test examples on training examples, without gradient updates.
- **Key properties:**
  - **In-context learning:** No fine-tuning required; training set is the "prompt."
  - **Bayesian prior:** Synthetic datasets sampled from a structural causal model prior; approximates Bayesian model averaging over a large class of generative models.
  - **Speed:** Single forward pass for inference; effectively "trains" in under 1 second.
  - **Constraint — classification only:** Original TabPFN outputs class probabilities; does not handle regression or survival natively.
  - **Constraint — small datasets:** Best performance on N < 1000 training samples, up to ~100 features. Performance degrades as N grows (context window fills; attention is O(N²)).
- **Reported performance:** Outperforms XGBoost, Random Forest, and other classical methods on a meta-dataset of small tabular classification benchmarks [verify: specific dataset numbers].
- **Extended versions (2024–2025):**
  - **TabPFN v2** (Hollmann et al., 2025 [verify]): extends to regression, larger datasets, improved prior. Published or preprinted around early 2025.
  - **TabICL** (Ye et al., 2024 [verify]): explores in-context learning for tabular data with different architectural choices.
- **Survival gap:** No native censoring-aware loss in TabPFN. Survival analysis requires handling right-censored labels, which standard cross-entropy loss ignores. Attaching a survival head requires either: (a) reformulating the loss with IPCW weights, (b) discretizing survival into classification bins and masking censored intervals, or (c) learning a new meta-training procedure with censored synthetic data. None of these have been published as of August 2025 [verify].

#### TabICL and Other 2024 Tabular FMs
- **TabICL:** Ye, M. et al. (2024). "In-context learning for tabular data: A survey and empirical study" or a specific model paper [verify: exact citation uncertain]. Explores prompt construction strategies for in-context tabular learning.
- **LIFT (Language-Interfaced Fine-Tuning):** Dinh, T. et al. (2022). "LIFT: Language-Interfaced Fine-Tuning with Expert Language Descriptions." *NeurIPS 2022*. Serializes tabular rows as natural language strings and fine-tunes LLMs — bridges tabular and language FMs.
- **CARTE** (Caron et al., 2024 [verify]): pretraining on heterogeneous tabular corpora.
- **TabTransformer** (Huang et al., 2020): transforms categorical embeddings via transformer, but numerical features left as-is.
- **FT-Transformer (Feature Tokenization Transformer):** Gorishniy et al. (2021). "Revisiting Deep Learning Models for Tabular Data." *NeurIPS 2021*. Embeds each feature (categorical or numerical) as a token, applies standard transformer attention. Strong baseline; forms the backbone of several later models including SurvTRACE.
- **GrowNet:** Badirli et al. (2020). Boosting shallow neural networks. Ensemble of weak learners (shallow NNs) combined with gradient boosting.
- **NODE (Neural Oblivious Decision Trees):** Popov, S., Morozov, S., & Babenko, A. (2020). "Neural oblivious decision trees for tabular data." *ICLR 2020*. Differentiable soft decision trees; layer-wise training; competitive with gradient boosted trees on some benchmarks.

---

### 3.2 Applications to Clinical/EHR Data

#### Self-Supervised Pretraining on EHR Tables
- **ETHOS** [verify: citation uncertain]: EHR pretraining with survival-aware objectives.
- **MedBERT / ClinicalBERT on structured data:** Most clinical BERT models operate on clinical notes (unstructured), not tabular EHR features. Adapting them to tabular survival data requires feature tokenization.
- **SCARF** (Bahri et al., 2022, *ICLR*): Self-Supervised Contrastive Learning using Random Feature Corruption for tabular data. Randomly corrupts features (replacing with random values from the marginal distribution), then trains a contrastive encoder to distinguish original from corrupted. Shown to improve downstream performance with limited labels. Applicable to EHR pretraining in principle.
- **SubTab** (Ucar et al., 2021, *NeurIPS*): Splits features into subsets, reconstructs full feature space from subsets via contrastive learning. Designed for tabular self-supervised learning.
- **HAIM** (Soenksen et al., 2022, *Nature Medicine* [verify]): Holistic AI in Medicine — combines multimodal EHR data (labs, vitals, notes, imaging) into a unified representation for outcome prediction, including survival-type outcomes. Shows that multimodal integration improves predictions.
- **Tabular SSL on MIMIC-III/IV:** Several groups have applied masked autoencoder-style pretraining to MIMIC vital signs and lab tables [verify: specific papers]. Common finding: self-supervised pretraining helps when labeled data is scarce (<500 events).
- **Direct survival applications:** Very few papers combine tabular foundation model pretraining with survival-specific losses. SurvTRACE (Section 2.2) is the closest — in-domain transformer pretraining + survival fine-tuning. A foundation model pretrained across multiple survival datasets and then fine-tuned on a target dataset has not been published as of August 2025 [verify].

---

## 4. Transformers & Attention for Survival Analysis

### 4.1 SurvivalBERT / Clinical BERT Approaches
- **ClinicalBERT** (Huang et al., 2019; Alsentzer et al., 2019): BERT pretrained on clinical notes (MIMIC-III). Used for downstream clinical tasks including mortality prediction, but predominantly from unstructured notes. Not tabular survival analysis.
- **Survival with BERT embeddings:** A common approach extracts ClinicalBERT embeddings from clinical notes, then feeds them into a Cox or DeepHit head for survival prediction. This is a feature extraction pipeline, not end-to-end survival pretraining.
- **MTSF (Medical Time Series Foundation models):** Several 2023–2024 papers pretrain transformers on time-series EHR (vital signs, labs over time) for mortality/readmission prediction. These are close to survival analysis but typically frame outcomes as binary classification rather than censored survival [verify].

### 4.2 Transformer-Based Survival Models (2022–2024)

- **SurvTRACE** (2022): Already covered in Section 2.2. Most directly relevant prior work for a TabPFN-survival project.
- **Transformer-based multi-event survival** (various 2022–2024 preprints): Multiple preprints on arXiv apply standard transformer architectures (often FT-Transformer backbone) to survival outcomes. Most are dataset-specific, without pretraining.
- **UniSurv / SurvivalFM** [verify: no confirmed paper found as of August 2025]: The concept of a foundation model pretrained across multiple survival datasets and capable of zero-shot or few-shot survival prediction appears to be an open research direction, not yet published.
- **MAMBA for survival** [verify]: State-space sequence models (Mamba, 2023) have been applied to EHR sequences; survival outcomes have been included as auxiliary tasks in some works, but dedicated survival-focused Mamba papers are sparse [verify].
- **Attention for competing risks:** Several 2023 papers propose attention mechanisms to weight the contribution of each feature to each cause-specific hazard separately. The intuition: different features are predictive for different causes of failure. Cross-cause attention (attending over cause-specific representations) is proposed in at least one preprint [verify].

### 4.3 Attention Mechanisms for Feature Importance in Survival
- **SHapley Additive exPlanations (SHAP) for survival:** SHAP has been extended to tree-based survival models (`shap` library supports `XGBSEDebiasedBCE` and RSF). For neural survival models, gradient-based attribution (SHAP DeepLIFT, integrated gradients) applies.
- **Attention weights as importance:** In SurvTRACE and similar transformer survival models, attention weights over feature tokens provide a form of feature importance. However, the relationship between attention weights and feature importance is contested in the interpretability literature (Jain & Wallace, 2019 [verify]).
- **Counterfactual and causal approaches:** Some survival papers (Alaa & van der Schaar, 2017 [verify]) frame survival prediction in a causal framework using attention-based propensity models for treatment effect estimation in survival settings.

---

## 5. Research Gaps & Opportunities

### Gap Analysis

- **[ ] Has anyone attached a survival head to TabPFN?**
  Almost certainly not, for the following structural reasons:
  1. TabPFN's meta-training objective is cross-entropy over multi-class synthetic classification problems. Survival analysis requires a censoring-aware loss (partial likelihood, IPCW Brier score, or discrete-time likelihood with masked censored intervals). The meta-training data generation pipeline would need to be completely redesigned to generate synthetic survival datasets with censoring mechanisms.
  2. TabPFN's in-context learning treats training labels as part of the input context. Censored labels are ambiguous — the model would need to be taught to distinguish censored from event labels and propagate the correct uncertainty.
  3. The constraint to N < 1000 may limit applicability to large survival cohorts (SEER, TCGA), though many clinical trial datasets fall within this range.
  4. **Opportunity:** Meta-train a "SurvPFN" on synthetic survival datasets (e.g., generated from parametric survival distributions with known covariates, random censoring mechanisms). At inference, the entire labeled+censored training set is the context, and the model predicts survival probabilities for test patients. This would be genuinely novel.

- **[ ] Has anyone used in-context learning for censored data?**
  No published work found as of August 2025 [verify]. ICL for survival/censored regression is a clear gap. The closest is TabICL for regression, but censoring is not addressed. Handling censored labels in-context requires the model to reason about partial information (we know the event has not yet occurred as of censoring time, not that it will not occur). This is conceptually distinct from missing-label semi-supervised learning.

- **[ ] Competing risks with foundation model embeddings?**
  Not found as of August 2025 [verify]. The combination of:
  (a) a foundation model for feature representation (e.g., pretrained tabular FM or LLM-based EHR encoder), plus
  (b) a competing-risks survival head (DeepHit-style or DSM-style),
  has not been systematically studied. SurvTRACE does in-domain pretraining + competing risks but is not a general FM. This is an achievable extension.

- **[ ] What's the state of calibration in deep survival models?**
  Calibration remains a significant open problem. Key findings from the literature:
  - Haider et al. (2020, *JMLR*): Systematic evaluation of calibration in deep survival models showed that most models (including DeepSurv, RSF) are substantially miscalibrated when evaluated with the D-calibration metric [verify: specific findings].
  - Calibration of competing risks models is even less studied.
  - Post-hoc recalibration methods (Platt scaling adapted for survival, isotonic regression on survival probabilities) exist but are rarely reported in deep learning survival papers.
  - IBS captures some calibration information but is dominated by discrimination in practice.
  - **Opportunity:** A model that is both well-calibrated and discriminative — possibly via a proper scoring rule loss (e.g., Brier score directly as training objective) — would be notable.

- **[ ] Are there open benchmarks comparing foundation model embeddings vs. hand-crafted features for survival?**
  No comprehensive, publicly released benchmark found as of August 2025 [verify]. Existing benchmarks:
  - **PyCox / `scikit-survival` benchmarks:** Compare classical methods and some deep learning models on SUPPORT2, METABRIC, GBSG, Rotterdam. No FM embeddings included.
  - **TCGA survival benchmarks:** Several papers use TCGA genomic data for survival, sometimes with FM-derived embeddings (e.g., protein LLM embeddings for gene expression). Not a standardized benchmark with multiple FM types.
  - **Opportunity:** A reproducible benchmark (code + data) comparing (1) raw features + classical models, (2) raw features + deep survival models, (3) FM embeddings + survival heads, across multiple datasets and evaluation metrics (C-index, IBS, D-calibration) would be a significant contribution and likely attract citations.

### Additional Gaps

- **[ ] Cross-dataset generalization / transfer learning for survival:**
  Can a model trained on one survival dataset (e.g., METABRIC) improve predictions on a related dataset (e.g., GBSG) via fine-tuning? Almost no systematic study of this exists for survival models, unlike in imaging or NLP [verify].

- **[ ] Few-shot survival prediction:**
  In rare disease settings, the number of observed events may be very small (< 50). TabPFN-style in-context learning could be particularly valuable here, but no work has exploited this.

- **[ ] Uncertainty quantification for survival:**
  Conformal prediction for survival (Candès group, 2023 [verify]) and Bayesian deep survival models exist but are not widely adopted. Connecting FM uncertainty estimates (e.g., ensemble of TabPFN-style in-context predictions) to survival uncertainty is unexplored.

---

## 6. Key Papers Reference Table

| Paper | Year | Model | Dataset(s) | C-index | Loss | Competing Risks? | Notes |
|-------|------|-------|------------|---------|------|-----------------|-------|
| Cox PH (Cox 1972) | 1972 | Semi-param Cox | Various | ~0.60–0.68 [verify] | Partial log-lik | No | Gold standard baseline |
| RSF (Ishwaran 2008) | 2008 | Random Survival Forest | Various | ~0.67–0.72 [verify] | Log-rank split | No | Strong non-parametric baseline |
| MTLR (Yu et al. 2011) | 2011 | Multi-task logistic reg. | Lung cancer [verify] | ~0.64–0.68 [verify] | Sequential logistic | No | Discretizes time |
| DeepSurv (Katzman 2018) | 2018 | Cox-NN (MLP) | SUPPORT2, METABRIC, GBSG | ~0.62–0.66 [verify] | Neg partial log-lik | No | Cox loss, no PH relaxation |
| DeepHit (Lee 2018) | 2018 | Discrete-time NN | METABRIC, SEER | ~0.69–0.71 [verify] | Likelihood + ranking | Yes | Standard CR baseline |
| PC-Hazard (Kvamme 2019) | 2019 | Piecewise-const hazard NN | METABRIC, SUPPORT2 | ~0.64–0.68 [verify] | Discrete-time lik | No | Relaxes PH assumption |
| Cox-Time (Kvamme 2019) | 2019 | Time-varying Cox NN | METABRIC, SUPPORT2 | ~0.63–0.65 [verify] | Sampled partial log-lik | No | Time-varying relative risk |
| N-MTLR (Fotso 2018) | 2018 | Neural MTLR | SUPPORT2 [verify] | ~0.64–0.66 [verify] | Sequential logistic (NN) | No | Neural extension of MTLR |
| NODE (Popov 2020) | 2020 | Neural oblivious DTs | Tabular benchmarks | N/A (no survival) | CE / MSE | No | Differentiable tree ensemble |
| TabNet (Arik 2021) | 2021 | Sparse attention tabular | UCI benchmarks | N/A (no survival) | CE | No | Instance-wise feature selection |
| DSM (Nagpal 2021) | 2021 | Mixture Weibull NN | SUPPORT2, METABRIC | ~0.67–0.70 [verify] | ELBO-based | Yes | Parametric, interpretable |
| SAINT (Somepalli 2021) | 2021 | Row+col attention | UCI benchmarks | N/A (no survival) | Contrastive + CE | No | Intersample attention |
| FT-Transformer (Gorishniy 2021) | 2021 | Feature token transformer | Tabular benchmarks | N/A (no survival) | CE / MSE | No | Backbone for SurvTRACE etc. |
| SurvTRACE (Wang 2022) | 2022 | Transformer + BERT pretrain | METABRIC, SEER | ~0.69–0.72 [verify] | Survival lik + masked feat | Yes | Most relevant prior work |
| TabPFN (Hollmann 2023) | 2023 | In-context learning transformer | UCI classification benchmarks | N/A (no survival) | CE (meta-trained) | No | N<1000, no censoring support |
| TabPFN v2 (Hollmann 2025 [verify]) | 2025 | In-context LM (extended) | Tabular regression+class | N/A (no survival) | CE + MSE | No | Extends to regression |
| XGBoost-Cox | 2016+ | Gradient boosted trees | Various | ~0.68–0.73 [verify] | Cox/AFT objective | No | Strong tree-based baseline |

---

## 7. Novel Directions — Ranked by Feasibility × Novelty

### 1. SurvPFN: In-Context Learning for Censored Survival Data (HIGH priority)
**Core idea:** Meta-train a TabPFN-style transformer on synthetic survival datasets with realistic censoring mechanisms (exponential, Weibull, administrative). At inference, provide the censored training set as context and predict test patients' survival probabilities (or full survival curves) without gradient updates.

**Novelty claim:** First in-context learning approach for censored survival analysis. Addresses both the "no fine-tuning" desiderata and the small-N regime common in clinical trials and rare disease settings.

**Technical approach:**
- Generate synthetic survival datasets: covariate matrix X ~ various distributions, event times T ~ Weibull/log-normal mixture with known β, censoring times C ~ exponential/uniform, observe (min(T,C), I(T≤C)).
- Meta-training objective: at each step, sample a synthetic dataset, randomly split into "context" (N_train) and "query" (N_test), minimize survival loss on query given context in a single forward pass.
- Survival loss options: (a) discrete-time negative log-likelihood (MTLR-style binning); (b) censoring-weighted Brier score; (c) partial likelihood approximation.
- Architecture: extend TabPFN with a causal masking scheme that distinguishes censored from event labels in the context.

**Feasibility:** Medium. Requires modifying TabPFN's meta-training pipeline (significant engineering), but the conceptual framework is clear. Synthetic data generation for survival is well-understood. Main risk: meta-learning may not generalize to real censoring patterns.

**Estimated C-index improvement potential:** +0.02–0.05 over DeepSurv/Cox on small datasets (N < 500) [speculative]; competitive with RSF on N < 200; main gains expected on few-shot regimes.

**Datasets for evaluation:** SUPPORT2, METABRIC, GBSG, Rotterdam (split into small subsets), rare disease datasets if available.

---

### 2. Competing Risks SurvPFN with Cause-Specific Heads (HIGH priority, extension of #1)
**Core idea:** Extend SurvPFN to competing risks by adding K cause-specific output heads (analogous to DeepHit's multi-cause architecture) while keeping the in-context learning backbone.

**Novelty claim:** First in-context learning approach for competing risks. Combines the statistical rigor of DeepHit-style modeling with the data-efficiency of in-context learning.

**Technical approach:**
- Synthetic competing risks data generation: simulate K causes with cause-specific Weibull distributions; the observed event is min over K event times; generate cause-specific censoring via independent competing mechanisms.
- Loss: cause-specific discrete-time likelihood with IPCW weighting for censored observations.
- Architecture: single shared transformer context encoder, K cause-specific MLP heads.

**Feasibility:** Medium-Hard (builds on #1; requires additional cause-label handling in meta-training). If #1 is implemented, this is a natural extension with ~2–4 additional weeks of work.

**Estimated gain:** On METABRIC competing risks: [verify baseline: DeepHit cause-1 C-index ~0.69]; target: +0.01–0.03 improvement.

---

### 3. Self-Supervised Tabular FM Fine-Tuned with Survival Loss (MEDIUM priority)
**Core idea:** Pretrain an FT-Transformer or SAINT encoder on a large corpus of tabular data (either general UCI/OpenML tables or a collection of survival datasets), then fine-tune with a survival-specific loss (DeepHit or PC-Hazard style) on the target dataset.

**Novelty claim:** Demonstrates that cross-dataset tabular FM pretraining provides measurable improvement over in-domain training for survival outcomes — extending SurvTRACE's in-domain pretraining to a true foundation model paradigm.

**Technical approach:**
- Pretraining corpus: collect 20–50 public survival datasets (TCGA sub-cohorts, SEER subsets, UCI survival datasets) or heterogeneous tabular data from OpenML.
- Pretraining objective: masked feature reconstruction (BERT-style on features) + optionally survival-aware objectives on labeled subsets.
- Fine-tuning: freeze encoder, train survival head; or end-to-end fine-tuning with reduced learning rate.
- Evaluation: compare against SurvTRACE (in-domain only), DeepHit, RSF.

**Feasibility:** Medium. Data collection and preprocessing across heterogeneous survival datasets is the main bottleneck. Training is straightforward given existing FT-Transformer/SAINT codebases.

**Estimated gain:** +0.01–0.03 C-index on low-resource target datasets; calibration improvement expected from more diverse pretraining [speculative].

---

### 4. Calibration-First Deep Survival Model via Proper Scoring Rules (MEDIUM priority)
**Core idea:** Train deep survival models directly on proper scoring rules (Brier score, log-likelihood with IPCW weighting) instead of Cox partial likelihood or ranking losses, and systematically evaluate calibration alongside discrimination.

**Novelty claim:** First systematic study of training objective choice on both C-index and calibration for deep survival models, with a proposed architecture that achieves Pareto improvement on discrimination + calibration jointly.

**Technical approach:**
- Implement DeepHit-style discrete-time model trained with (a) standard likelihood loss, (b) Brier score loss, (c) combined loss, (d) post-hoc calibration via isotonic regression / Platt scaling.
- Add temperature scaling adapted for survival curves (time-specific temperature).
- Evaluate with C-index, IBS, D-calibration, and time-dependent AUC across SUPPORT2, METABRIC, GBSG.

**Feasibility:** Easy-Medium. Does not require novel architecture; mostly an empirical study with targeted methodological contributions. Risk: findings may be "negative" (calibration and discrimination trade-offs are well-known).

**Estimated gain:** IBS improvement over DeepHit [verify: DeepHit IBS on SUPPORT2 ~0.18–0.22]; D-calibration improvement expected.

---

### 5. LLM-Based Serialization for Survival Prediction (LOWER priority for 3-month project)
**Core idea:** Serialize tabular patient records as natural language prompts (following LIFT / TabLLM paradigms) and use a pretrained LLM (e.g., LLaMA, GPT-4) with a survival-adapted output head or direct prompt-based risk scoring.

**Novelty claim:** Explores whether world knowledge encoded in LLMs (e.g., "smoking is a risk factor for lung cancer") provides survival prediction improvements over purely data-driven tabular models, especially in low-data regimes.

**Technical approach:**
- Serialize: "Patient is a 65-year-old female with stage III breast cancer, ER positive, grade 2 tumor, treated with chemotherapy." → LLM embedding → survival head.
- Compare: raw tabular features + survival model vs. LLM-serialized features + survival model.
- Datasets: METABRIC (rich clinical features), TCGA (genomic + clinical).

**Feasibility:** Medium (access to LLM API or local LLM required; serialization is dataset-specific). Risk: LLM embeddings may not outperform tabular FMs for structured clinical data; serialization loses some numerical precision; results may vary significantly with prompt engineering.

**Estimated gain:** Uncertain [verify — some TabLLM papers show modest gains in classification; survival transfer unclear].

---

## 8. Key Software & Datasets

### Software Libraries
- **`pycox`** (Kvamme et al.): Python library implementing DeepSurv, DeepHit, PC-Hazard, Cox-Time, N-MTLR. https://github.com/havakv/pycox
- **`scikit-survival`**: Scikit-compatible survival analysis (RSF, Cox, Lasso-Cox, C-index/IBS evaluation). https://scikit-survival.readthedocs.io/
- **`lifelines`**: Python survival analysis (KM, Cox, AFT models). https://lifelines.readthedocs.io/
- **`auton-survival`** (Carnegie Mellon): DSM, DRSA, and survival analysis utilities. https://github.com/autonlab/auton-survival
- **`xgboost`**: XGBoost with `survival:cox` and `survival:aft` objectives.
- **`survtrace`**: SurvTRACE implementation. https://github.com/RyanWangZf/SurvTRACE [verify: URL]
- **`TabPFN`**: https://github.com/automl/TabPFN

### Benchmark Datasets
- **SUPPORT2** (Study to Understand Prognoses Preferences Outcomes and Risks of Treatment): N ≈ 8873 ICU patients, 14 features, single event, moderate censoring. Widely used.
- **METABRIC** (Molecular Taxonomy of Breast Cancer International Consortium): N ≈ 1980 breast cancer patients, ~9 clinical features, two competing events (breast cancer death, other death). Standard competing risks benchmark.
- **GBSG** (German Breast Cancer Study Group): N ≈ 2232, 8 features, single event (recurrence). Classic benchmark.
- **Rotterdam**: N ≈ 2982 breast cancer patients, two events. Often used with GBSG for cross-validation experiments.
- **SEER** (Surveillance, Epidemiology, and End Results): Large-scale cancer registry; used in DeepHit, SurvTRACE. Multiple competing causes. N > 100K in some extractions.
- **TCGA** (The Cancer Genome Atlas): Multi-cancer genomic + clinical survival data. Used for deep learning + genomics survival papers.
- **FLCHAIN** (Free Light Chain study): N ≈ 7874, clinical features, single event. Available in `lifelines`.
- **UNOS/NHANES derived datasets**: Sometimes used for cardiovascular survival benchmarks.

---

## References (Selected, Alphabetical by First Author)

Alaa, A.M. & van der Schaar, M. (2017). Deep multi-task gaussian processes for survival analysis with competing risks. *NeurIPS*.

Arik, S.Ö. & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *AAAI*.

Austin, P.C. et al. (2020). Graphical calibration curves and the integrated calibration index (ICI) for survival models. *Statistics in Medicine*.

Bahri, D. et al. (2022). SCARF: Self-supervised contrastive learning using random feature corruption. *ICLR*.

Binder, H. & Schumacher, M. (2008). Allowing for mandatory covariates in boosting estimation of sparse high-dimensional survival models. *BMC Bioinformatics*.

Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.

Cox, D.R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*.

Fotso, S. (2018). Deep neural networks for survival analysis based on a multi-task framework. arXiv:1801.05512.

Gorishniy, Y. et al. (2021). Revisiting deep learning models for tabular data. *NeurIPS*.

Graf, E. et al. (1999). Assessment and comparison of prognostic classification schemes for survival data. *Statistics in Medicine*.

Haider, H. et al. (2020). Effective ways to build and evaluate individual survival distributions. *JMLR*, 21(85).

Harrell, F.E. et al. (1982). Evaluating the yield of medical tests. *JAMA*.

Heagerty, P.J. & Zheng, Y. (2005). Survival model predictive accuracy and ROC curves. *Biometrics*.

Hollmann, N. et al. (2023). TabPFN: A transformer that solves small tabular classification problems in a second. *ICLR*.

Ishwaran, H. et al. (2008). Random survival forests. *Annals of Applied Statistics*.

Kaplan, E.L. & Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*.

Katzman, J.L. et al. (2018). DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Medical Research Methodology*.

Kvamme, H. et al. (2019). Time-to-event prediction with neural networks and Cox regression. *JMLR*.

Kvamme, H. & Borgan, Ø. (2019). Continuous and discrete-time survival prediction with neural networks. arXiv:1910.06724.

Lee, C. et al. (2018). DeepHit: A deep learning approach to survival analysis with competing risks. *AAAI*.

Nagpal, C. et al. (2021). Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks. *JMLR*.

Popov, S. et al. (2020). Neural oblivious decision trees for tabular data. *ICLR*.

Royston, P. & Sauerbrei, W. (2004). A new measure of prognostic separation in survival data. *Statistics in Medicine*.

Simon, N. et al. (2011). Regularization paths for Cox's proportional hazards model via coordinate descent. *Journal of Statistical Software*.

Somepalli, G. et al. (2021). SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training. arXiv:2106.01342.

Tibshirani, R. (1997). The lasso method for variable selection in the Cox model. *Statistics in Medicine*.

Wang, Z. et al. (2022). SurvTRACE: Transformers for survival analysis with competing events. *CHIL*.

Yu, C.N. et al. (2011). Learning patient-specific cancer survival distributions as a sequence of dependent regressors. *NeurIPS*.

---

## 8. Retrieval-Augmented Survival Analysis

### 8.1 Background: Retrieval Augmentation in In-Context Learning

Retrieval-augmented generation (RAG) was popularized in the NLP setting (Lewis et al., 2020, *NeurIPS*: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"), where retrieved documents are prepended to the LLM's context window. The core motivation directly applies to TabPFN: the in-context learning mechanism is only as good as the examples it conditions on. If the full training set N >> context capacity, or if most training examples are irrelevant to a given test point, selecting K nearest neighbors as context should improve predictions over random or uniform sampling.

**Core question for SurvPFN:** Is retrieval-augmented TabPFN for survival **novel**? The answer, as of August 2025, is **yes — novel at the intersection** — though adjacent work exists in each individual sub-area.

---

### 8.2 Retrieval-Augmented Tabular Learning

#### Has RAG been applied to TabPFN?
No published paper as of August 2025 applies explicit retrieval augmentation (nearest-neighbor context selection) to TabPFN for **any** task [verify]. However, two adjacent developments exist:

- **TabPFN's implicit retrieval:** TabPFN's cross-attention over the training context is conceptually similar to a soft retrieval — the model attends to training examples that are most relevant to each test point. However, this is a soft, learned attention, not hard retrieval, and is limited by context window size.
- **TabICL context selection (Ye et al., 2024 [verify]):** Some TabICL and related ICL-for-tables papers study which training examples to include in the context prompt. Results suggest that semantically similar examples (in feature space) produce better ICL predictions than random samples, but this is studied in the classification setting without censoring. Specific paper: possibly "In-Context Learning for Tabular Data" or a related preprint — **exact citation uncertain [verify]**.
- **RETRIEVALTAB / KATE-style retrieval for tabular ICL:** In the NLP literature, KATE (Liu et al., 2022, "What Makes Good In-Context Examples for GPT-3?") demonstrates that retrieving k-nearest exemplars by embedding similarity dramatically improves GPT-3 few-shot performance. Direct analogs for non-language ICL tabular models are not published as of August 2025 [verify].

#### k-NN Augmented Tabular Models
- **SAINT intersample attention (Somepalli et al., 2021):** SAINT's row attention mechanism is structurally similar to soft k-NN — the model learns to attend to similar training rows. However, it operates over mini-batches (not dynamic retrieval), uses no explicit similarity metric, and is trained end-to-end rather than applying a pre-defined retrieval step. Not k-NN augmentation per se.
- **TabR (Gorishniy et al., 2023 [verify]):** "TabR: Tabular Deep Learning Meets Nearest Neighbors." This is the **most directly relevant paper** to retrieval-augmented tabular learning. TabR explicitly retrieves k nearest neighbors from the training set at inference time and uses their labels as additional context for a neural network. Key details:
  - Retrieval metric: Euclidean distance in raw feature space (with learned per-feature weights [verify]).
  - Architecture: Linear layer producing a query embedding; retrieved training samples (features + labels) are aggregated via attention and fused into the main prediction.
  - Datasets: standard tabular benchmarks (California Housing, Adult, Higgs, Jannis, etc.) — **no survival datasets**.
  - Reported gains: Competitive with XGBoost and FT-Transformer; in some settings outperforms pure attention-based models by using exact label retrieval.
  - **Key gap:** No censoring-aware handling; no survival outcomes; no analysis of censored label retrieval.
- **k-NN as a baseline:** k-NN with Euclidean distance remains a standard tabular baseline. It is not typically framed as "retrieval augmentation" — it *is* the model. TabR bridges k-NN and neural networks.

#### Retrieval in Non-Language ICL Settings
- **Prototype networks (Snell et al., 2017, *NeurIPS*):** Few-shot classification via class prototype construction. Conceptually similar — retrieve class exemplars, classify by distance to prototypes. No tabular or survival application.
- **Algorithmic alignment (Abbe et al., 2023 [verify]):** Theory of when ICL models learn to implement specific algorithms (including k-NN); relevant for understanding why nearest-neighbor context selection should work for TabPFN.

---

### 8.3 Nearest-Neighbor Survival Models

This area has a longer history than retrieval-augmented tabular learning.

#### k-NN Kaplan-Meier (k-NNKM)
- **Concept:** For a test point x, identify the k most similar training samples by feature distance, then compute the Kaplan-Meier estimate on that local neighborhood. This produces an individualized survival curve conditioned on the local feature distribution.
- **Key papers:**
  - **Beran (1981):** "Nonparametric regression with randomly censored survival data." Technical Report, UC Berkeley. The earliest formal treatment of nonparametric conditional survival estimation — essentially kernel/k-NN smoothing of the survival function. This is the theoretical ancestor of k-NNKM.
  - **Gefeller & Dette (1992 [verify]):** Bandwidth selection for kernel-smoothed survival curves.
  - **Lowdin & Harrell [verify]:** Various authors have described k-NN KM in textbook treatments.
  - **More recent:** Steingrimsson et al. (2019, *Biostatistics* [verify]): "Censoring unbiased regression trees and ensembles" — local tree-based methods for survival prediction that are conceptually related.
- **Similarity metric:** Euclidean distance on raw features, or Mahalanobis distance. Feature scaling is critical.
- **Improvements over KM baseline:** Substantial on heterogeneous populations; directly inherits the nonparametric consistency of KM in the local neighborhood under appropriate assumptions.
- **Censoring handling:** Censoring is handled implicitly within the KM estimator applied to the retrieved neighborhood — standard Kaplan-Meier handles right censoring correctly. No special treatment needed for the retrieval step itself.
- **Limitation:** Performance degrades in high-dimensional feature spaces (curse of dimensionality); does not learn feature representations; no competing risks extension in standard formulation.

#### Conditional Kaplan-Meier / Local Survival Regression
- **Dabrowska (1989):** Kernel-based nonparametric conditional survival function estimation. Formal asymptotic theory. Uses a kernel function K((x - x_i)/h) as similarity weights rather than hard k-NN selection.
- **Li & Datta (2001 [verify]):** Bootstrap-based confidence intervals for kernel-smoothed survival functions.
- **Random Survival Forests as retrieval:** RSF terminal nodes act as implicit retrievals — a test point falls into a terminal node, and the Nelson-Aalen estimate in that node uses only the training samples that co-occur in the leaf. This is algorithmically a weighted retrieval, though not framed that way in the literature.

#### Instance-Based / Lazy Learning for Survival
- **Lowdin & Harrell "lazy learning" in survival [verify: specific citation]:** The term "lazy learning" (cover & Hart, 1967 for k-NN classification) has been applied to survival analysis, but no major dedicated paper treats lazy survival learning as a primary contribution.
- **Survival-weighted k-NN (Molinaro et al., 2003 [verify]):** "Tree-based multivariate regression and density estimation with right-censored data." Uses tree-based distance for k-NN survival. The tree defines which samples are "similar."
- **Random forests for non-parametric KM estimation:** RSF's leaf-based KM is perhaps the most widely deployed form of local survival estimation. It is theoretically a retrieval model (retrieve samples in the same leaf) with a learned similarity metric (the forest's split structure).

#### Matching-Based Causal Survival Estimation
- **Matching in causal survival:** Matching estimators for treatment effects on survival outcomes are extensively studied:
  - **Rosenbaum & Rubin (1983):** Propensity score matching — retrieve matched controls with similar propensity scores for each treated patient; compute treatment effect on survival.
  - **Ho et al. (2011, *Journal of Statistical Software*):** `MatchIt` — practical matching for observational causal inference including survival outcomes.
  - **Abadie & Imbens (2006, *Econometrica*):** Large-sample properties of matching estimators for average treatment effects.
  - **Imai et al. (2014 [verify]):** Matching for causal survival with competing risks.
- **Difference from our setting:** Matching for causal survival is designed to balance covariates between treatment groups, not to build a predictive survival model. The retrieval is over controls, not training exemplars for prediction.

---

### 8.4 Retrieval-Augmented Learning for Clinical Prediction

#### Retrieval with EHR Prediction
- **DKNN for clinical predictions [verify]:** Deep k-nearest neighbors (Papernot & McDaniel, 2018, "Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning") applies k-NN in deep feature space for uncertainty estimation. Applications to clinical classification have been explored (mortality, readmission), but not survival analysis.
- **Case-based reasoning in medicine:** Long history (1990s–2000s) of case-based reasoning (CBR) for clinical diagnosis — retrieve similar past cases, adapt their outcomes. Survival prediction via CBR was studied in oncology (e.g., Lenz et al. [verify]), but predates modern deep learning and retrieval strategies. Relevant conceptually but technically dated.
- **MedRetrieval / clinical RAG (2023–2024):** Several papers apply RAG to LLM-based clinical notes analysis (e.g., retrieving similar clinical notes for differential diagnosis). These apply to unstructured text, not tabular survival data, and are conceptually distant.
- **Similarity-weighted survival curves (Gao et al. 2022 [verify: exact citation uncertain]):** A concept explored in some oncology informatics papers where the survival curve for a new patient is constructed as a weighted average of training patient survival curves, with weights proportional to covariate similarity. This is essentially retrieval-augmented KM. No major ML venue publication confirmed; may exist in biostatistics/bioinformatics venues [verify].

#### Relevant Negative Result
- **RETAIN (Choi et al., 2016, *NeurIPS*):** Reverse-time attention model for EHR prediction — uses attention over past visits to predict future events. Not retrieval-based (attends over a patient's own history, not other patients). Mortality prediction, not survival analysis.

---

### 8.5 RAG + In-Context Learning Theory

#### Does Retrieval Improve ICL?
Yes — substantial evidence from the NLP literature:

- **KATE (Liu et al., 2021 [verify: exact year]):** k-Nearest Neighbor Augmented in-conTExt learning. Retrieves training examples similar to the test input by embedding cosine similarity, prepends them to GPT-3's context. Shows consistent improvement over random example selection across multiple NLP tasks (classification, QA). Key finding: relevant retrieval more important than diverse retrieval for classification; diverse retrieval helps for generation tasks.
- **Rubin et al. (2022, *ACL*):** "Learning To Retrieve Prompts for In-Context Learning." Trains a retriever specifically optimized for ICL performance (rather than using a fixed similarity metric). Shows that task-specific retrieval significantly outperforms static similarity retrieval.
- **Su et al. (2022 [verify]):** "Selective Annotation Makes Language Models Better Few-Shot Learners." Shows that selecting diverse, representative examples via graph-based methods outperforms similarity-only selection for some tasks.
- **Zhang et al. (2023 [verify]):** "In-Context Learning with Retrieved Demonstrations for Conversational Tasks." Confirms retrieval superiority over random selection.

#### Optimal Context Selection Strategies
The literature identifies several strategies (roughly in order of effectiveness for tabular/structured settings):

1. **Embedding similarity (cosine / Euclidean):** Most common baseline. Use a pretrained encoder to embed both training and test examples; retrieve top-K by similarity. Outperforms random selection consistently.
2. **BM25 (sparse lexical retrieval):** Effective for natural language. Less directly applicable to numeric tabular data; could be applied to serialized representations.
3. **Diverse retrieval (Maximum Marginal Relevance, DPP-based):** Retrieve a diverse set of K examples that covers the input space near the test point. Helps when K is large or the model benefits from seeing varied examples. MMR (Carbonell & Goldstein, 1998) balances relevance and diversity.
4. **Learned retrievers:** End-to-end trained retrieval (REALM, RAG, RETRO in NLP). For tabular settings, training a retriever jointly with the ICL predictor is unexplored as of August 2025 [verify].
5. **Label-stratified retrieval:** Ensure retrieved examples cover the label distribution (relevant for survival: retrieve examples with short, medium, and long event times). Specifically useful when the label distribution is imbalanced by censoring.

#### Theoretical Justification for Tabular ICL Retrieval
- **Akyürek et al. (2022, "What learning algorithm is in-context learning?"):** Shows that transformers in ICL implicitly implement gradient descent or k-NN classification on the provided context. If the model is implicitly doing k-NN, then providing the k most relevant training examples (true k-NN) should improve alignment between the model's implicit algorithm and the optimal algorithm for the test distribution.
- **Implication for TabPFN:** TabPFN is meta-trained to approximate Bayesian inference over the training context. Providing the most predictively relevant training examples (nearest neighbors in feature space) should better approximate the posterior over the local feature distribution, yielding better-calibrated predictions.

---

### 8.6 Summary: Novelty Assessment for Retrieval-Augmented TabPFN Survival

| Axis | Status |
|------|--------|
| RAG applied to TabPFN (any task) | **Not published** as of August 2025 [verify] |
| RAG applied to TabPFN for survival | **Not published** — clearly novel |
| k-NN survival estimation (classical) | Well-studied (Beran 1981, k-NNKM) |
| Neural k-NN for tabular (non-survival) | TabR (Gorishniy 2023) is the closest work |
| Retrieval for clinical survival prediction | Concept explored informally; no major ML paper [verify] |
| RAG improving ICL (NLP) | Well-established (KATE, Rubin et al., etc.) |
| Censoring-aware retrieval | No published work found [verify] |

**Novelty verdict:** Retrieval-augmented TabPFN for survival is **novel at the intersection**. The closest prior art is TabR (neural k-NN for tabular, no survival), k-NNKM (k-NN survival, no neural learning), and KATE (retrieval for NLP ICL). None of these three lines have been connected.

**Recommended first retrieval strategy:** Euclidean distance in raw (standardized) feature space, matching the metric used in k-NNKM. Simple, fast, interpretable, and directly comparable to the classical k-NNKM baseline. Second variant: learned embedding similarity from a pretrained FT-Transformer or TabPFN's own internal representations.

**Key design question:** Should censored training samples be retrieved? Yes — censored samples still carry information (we know the patient survived at least until censoring time). The survival model must handle the censored labels in context; excluding censored samples would artificially distort the retrieved distribution, especially at short censoring times. The retrieval strategy itself need not be censoring-aware (retrieve by feature similarity); the survival model handles censoring in the likelihood.

---

## 9. Meta-Training TabPFN for Survival: Feasibility & Time Estimate

### 9.1 How TabPFN Generates Synthetic Data

Understanding the original TabPFN meta-training pipeline is prerequisite to estimating the effort to extend it to survival.

#### The Prior Over Data-Generating Processes
TabPFN's key insight: instead of training on real datasets, train on a **prior over synthetic datasets** that approximates the distribution of real-world tabular classification problems. The prior is a **structural causal model (SCM) prior**, described in detail in:

- Hollmann et al. (2023, *ICLR*), and supporting material in the precursor work:
- **Prior-Data Fitted Networks (PFN, Müller et al., 2022, *ICLR*):** "Transformers Can Do Bayesian Inference." Introduced the meta-learning framework; TabPFN specializes the prior to tabular classification.

**Key components of the TabPFN synthetic data prior:**

1. **Bayesian network structure:** Random DAGs are sampled to define causal relationships among features. Edge density and graph depth are sampled from distributions calibrated to real-world datasets.
2. **Node distributions:** Each node's conditional distribution is sampled from a mixture of: linear functions, non-linear transformations (e.g., neural network layers with random weights), categorical splits. This produces heterogeneous feature distributions.
3. **Label generation:** A "label function" is applied to a subset of features (or a learned representation): class probabilities are produced by a randomly-sampled neural network applied to the feature vector. Multiple classes are supported.
4. **Dataset scale:** Each synthetic dataset has N_train ∈ [10, 1000] samples (approximately) and D ∈ [1, 100] features. The number of datasets sampled during meta-training is ~3,000,000 (3M), though many are discarded if trivially easy or degenerate.
5. **Noise:** Label noise (random label flipping) and feature noise are added at varying rates to simulate real-world data quality.

**What makes this prior effective:** The SCM structure means synthetic datasets have realistic covariate correlation structures (unlike i.i.d. feature generation). The random neural network label functions produce non-linear, non-monotonic relationships that approximate the distribution of real tabular classification problems.

**Code reference:** The prior is implemented in `tabpfn/priors/` in the TabPFN GitHub repo, primarily in `tabpfn/priors/prior_bag.py` and `tabpfn/priors/gp_mix.py` [verify: exact file names may differ across versions].

---

### 9.2 Modifications Required for Survival Meta-Training

To meta-train a "SurvPFN" (SurvivalPFN) on synthetic censored survival datasets, the following changes to the data generation pipeline are required:

#### Step 1: Replace Label Generation with Survival Time Generation
Instead of generating class labels (0/1/…/K), generate survival times T from a parametric survival distribution conditioned on features:

```
T_i | x_i ~ Weibull(α(x_i), β(x_i))   or   log-normal(μ(x_i), σ(x_i))
```

where α(x_i) and β(x_i) are produced by a randomly-sampled neural network applied to x_i (paralleling TabPFN's random label function). A mixture of Weibull distributions is more flexible and covers a broader range of hazard shapes (monotone increasing, decreasing, bathtub).

**Effort:** Low — parametric survival time generation is straightforward. The main choice is the distribution family and how to parameterize the covariate effect. Log-linear link functions (log(α) = β^T x) mimic Cox PH; non-linear mappings relax this.

#### Step 2: Introduce Censoring Mechanisms
Generate censoring times C_i independently of T_i (right censoring assumption):

- **Exponential censoring:** C_i ~ Exponential(λ_C), where λ_C is sampled per-dataset from a distribution calibrated to produce realistic censoring rates (20%–80%).
- **Administrative censoring:** C_i = t_admin (fixed study end time), varies per dataset.
- **Mixture censoring:** C_i = min(Exp(λ), t_admin) — more realistic.

Observed outcome: (Y_i, δ_i) = (min(T_i, C_i), I(T_i ≤ C_i)).

**Key calibration:** The synthetic prior must be calibrated so that the meta-training distribution of censoring rates and event time distributions matches the distribution encountered at real-data inference time. This calibration is non-trivial and likely requires empirical validation on held-out real survival datasets.

**Effort:** Low-Medium — implementing the censoring mechanism is simple; calibrating the censoring rate distribution requires some experimentation.

#### Step 3: Modify the Meta-Training Loss
The classification cross-entropy loss must be replaced with a survival-appropriate loss. Options (in increasing complexity):

**Option A: Discrete-time negative log-likelihood (MTLR-style)**
- Discretize time into K bins (e.g., K=20–100 quantile-based bins per dataset).
- Predict a probability vector P(T ∈ bin k | x) for each test point.
- Loss: for event observations, -log P(T ∈ bin_k | x); for censored observations, -log P(T > C | x) = -log Σ_{j > k_C} P(T ∈ bin_j | x).
- **Advantage:** Clean multi-class softmax output (K classes instead of C classes); directly generalizes TabPFN's classification output.
- **Implementation change:** Output head: replace C-class softmax with K-bin softmax (K fixed at, say, 50). Loss computation: add censoring-aware masking for censored samples. Time bins: determined per dataset from the synthetic event time distribution.
- **Effort:** Medium — output head modification + loss modification are contained changes.

**Option B: Proportional hazards loss (partial likelihood)**
- Predict a scalar log-risk h(x) per patient (like DeepSurv).
- Loss: Cox partial log-likelihood computed over the synthetic dataset.
- **Advantage:** Scalar output, simpler head.
- **Disadvantage:** Only predicts risk ordering, not survival curves; harder to output calibrated survival probabilities at inference.
- **Effort:** Medium — requires implementing batched partial likelihood inside the meta-training loop; O(N²) naive implementation.

**Option C: Continuous-time survival via neural ODE or parametric head [verify]**
- Predict parameters (α(x), β(x)) of a Weibull survival function directly.
- Loss: exact Weibull log-likelihood for events, survival function for censored.
- **Advantage:** Smooth survival curves; extrapolates beyond observed time.
- **Effort:** Medium-High — requires numerical stability in Weibull parameterization; may mismatch when real data is not Weibull.

**Recommended:** Option A (discrete-time NLL) for the first implementation, as it most naturally extends TabPFN's multi-class output while correctly handling censoring.

#### Step 4: Modify the Transformer Architecture
The core TabPFN transformer architecture requires modifications to understand that context labels are (time, event_indicator) pairs rather than class labels:

- **Context label encoding:** Instead of a class embedding (integer → embedding), encode the context label as (log(Y_i), δ_i) → a 2D or higher-dimensional embedding via a learned linear projection. This allows the model to distinguish censored from event observations.
- **Output head:** Replace the K-class softmax head with a K-bin survival softmax head.
- **Positional/time encoding:** The K time bins must be communicated to the model; one approach is to include a fixed time-bin embedding in the output query.

**Effort:** Medium — modifying the label encoding requires touching the architecture. The rest of the transformer (attention mechanism, positional encodings for features) can be reused.

#### Step 5: Meta-Training Logistics
- **Number of synthetic datasets:** Original TabPFN uses ~3M. For SurvPFN, a similar number is likely needed for generalization. With simpler survival tasks, fewer may suffice (~500K–1M) [speculative; verify empirically]. Start with 100K for initial experiments.
- **Batch size:** Same as TabPFN — multiple datasets per batch (each dataset = one meta-training example).
- **Compute:** See Section 9.4 below.

---

### 9.3 Existing Work on Meta-Learning for Survival

#### Has Anyone Meta-Trained on Synthetic Survival Tasks?
No published paper as of August 2025 applies PFN-style meta-training (prior-data fitted networks, synthetic dataset generation) to survival analysis [verify]. This is a clear gap.

**Adjacent work:**

- **Few-shot survival / transfer learning for survival (various 2022–2024 [verify]):** Some papers fine-tune survival models across datasets (e.g., GBSG → METABRIC transfer), but none use meta-training on synthetic data.
- **MAML for survival [verify]:** Model-Agnostic Meta-Learning (Finn et al., 2017) has been applied to medical prediction tasks in a few papers, but dedicated survival applications with censoring-aware adaptation are not found as of August 2025 [verify].
- **Prototypical networks for censored regression [verify]:** No found paper.
- **Amortized inference for survival:** Variational autoencoders applied to survival (e.g., survival VAE) exist, but are not meta-trained across multiple datasets.
- **PFN extensions (non-survival):** RealTabFormer, TabICL, and other 2023–2024 PFN-adjacent work extend in-context learning for tabular regression and mixed outcomes, but none handle censoring.

#### Few-Shot Survival: The Motivating Use Case
The most compelling use case for meta-training is the **few-shot regime** (N_train < 200 events). In rare diseases (e.g., rare sarcomas, pediatric cancers, uncommon autoimmune conditions), this is the norm rather than the exception. Classical methods and deep survival models both struggle here; meta-trained SurvPFN could provide reliable survival estimates by leveraging the distributional prior learned from millions of synthetic datasets.

---

### 9.4 Technical Feasibility and Time Estimates

#### Engineering Time to Modify Synthetic Data Generation

| Task | Complexity | Estimated Person-Days |
|------|-----------|----------------------|
| Understand existing TabPFN prior code | Low | 3–5 days |
| Implement parametric survival time generation (Weibull/log-normal) | Low | 2–3 days |
| Implement censoring mechanisms (exponential, administrative) | Low | 2–3 days |
| Calibrate prior to match real survival dataset statistics | Medium | 5–7 days |
| Modify label encoding in transformer (event time + indicator) | Medium | 3–5 days |
| Implement discrete-time survival loss (Option A) | Medium | 4–6 days |
| Modify output head (K-bin softmax instead of C-class) | Low | 2–3 days |
| End-to-end meta-training loop integration | Medium | 5–7 days |
| Debugging and initial validation on synthetic holdout | Medium | 5–10 days |
| **Total engineering (prior + architecture + loss)** | | **~31–49 person-days** |

With 1 engineer: **6–10 weeks** of focused engineering.
With 2 engineers (parallelizing prior work and architecture work): **4–6 weeks**.

#### Training Time (GPU Hours)

Benchmarks from TabPFN training [verify: estimates are approximate]:
- Original TabPFN (3M synthetic datasets, classification): trained on 4–8 A100 GPUs for approximately 24–48 hours [verify: specific compute not reported in the paper; estimate from community reproductions].
- **SurvPFN estimate (1M synthetic datasets, survival):**
  - Survival datasets are similar in size/complexity to classification datasets.
  - The main additional cost: computing the survival loss (censoring-aware NLL) vs. cross-entropy — modest overhead (~10–20%).
  - Estimated training time: **24–72 GPU-hours on 4× A100** for 1M synthetic datasets [speculative; verify].
  - For 3M datasets: **72–200 GPU-hours on 4× A100** [speculative].
- **Cloud cost estimate (AWS/GCP A100 ~$3–4/hr per GPU, 4 GPUs):** $12–16/hr → ~$300–3200 for full training run depending on scale [verify current pricing].
- **Practical:** Multiple short training runs (100K, 500K, 1M datasets) to validate learning before committing to full-scale training.

#### Main Technical Failure Modes

1. **Prior mismatch (most likely failure mode):** The synthetic survival data distribution may be too far from real survival datasets. Specifically: real survival times often have heavy tails (long survivors), complex covariate interactions, and dataset-specific censoring patterns. If the synthetic prior misses these, the meta-trained model will not generalize. **Mitigation:** Calibrate prior parameters against moments of real survival datasets (event rate, censoring rate, tail behavior).

2. **Censoring mechanism learning:** The model must learn to correctly propagate uncertainty from censored context examples to test predictions. If censoring is too heavy (>70% censoring) or too light (<10%), the model may degenerate. **Mitigation:** Sample censoring rate uniformly over [0.1, 0.9] during meta-training.

3. **Time discretization sensitivity:** The K time bins are synthetic-dataset-specific. At inference on real data, the mapping from the model's bin indices to real times must be handled carefully. **Mitigation:** Use quantile-based binning (e.g., bin boundaries at event time quantiles) communicated as part of the input context — the model sees bin boundaries as additional inputs.

4. **Context size vs. dataset size:** Real survival datasets have N >> 1000 (SEER: N > 100K). Original TabPFN degrades for N > 1000 due to O(N²) attention. **Mitigation:** Use retrieval augmentation (Section 8) to select a context of K = 100–500 most similar training samples; this is a natural complement to SurvPFN and directly motivates the joint retrieval + meta-training direction.

5. **Competing risks extension:** Adding competing risks (K causes) multiplies output dimensionality and loss complexity. **Mitigation:** Implement single-risk SurvPFN first; add competing risks as a subsequent extension.

6. **Evaluation gap between synthetic and real performance:** The meta-training loss (on synthetic survival tasks) may not correlate with C-index/IBS on real datasets. **Mitigation:** Use a meta-validation set drawn from real survival datasets to monitor generalization.

#### Total Calendar Time Estimate

**With 1 engineer:**
- Weeks 1–2: Codebase familiarization + prior modification (survival time + censoring generation)
- Weeks 3–4: Architecture modification (label encoding + output head) + loss implementation
- Weeks 5–6: Integration + debugging + initial small-scale runs (100K synthetic datasets)
- Weeks 7–8: Calibration of prior + medium-scale training (500K–1M)
- Weeks 9–10: Evaluation on real survival datasets + comparison with baselines
- Weeks 11–12: Competing risks extension or retrieval augmentation (if time permits)
- **Total: 10–12 weeks → feasible within 3 months.**

**With 2 engineers:**
- One engineer handles prior modification + data generation pipeline (Weeks 1–4).
- Second engineer handles architecture + loss modifications (Weeks 1–4).
- Joint integration + training + evaluation (Weeks 5–8).
- Competing risks + retrieval extension (Weeks 9–10).
- **Total: 8–10 weeks → comfortably within 3 months.**

**Critical path item:** Training time is not the bottleneck (24–72 GPU-hours is manageable). Engineering and debugging are the bottleneck. The prior calibration step (ensuring synthetic data matches real survival distributions) is likely the hardest and most uncertain step.

**Feasibility verdict:** **Yes, feasible within 3 months** for a single-risk SurvPFN. Competing risks extension feasible within 3 months only with 2 engineers. Retrieval-augmented SurvPFN (combining Sections 8 and 9) is feasible within 3 months as a joint contribution with focused scope.

---

### 9.5 Recommended Implementation Roadmap

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Prior + Loss | 2–3 weeks | Synthetic survival dataset generator; discrete-time NLL loss implemented |
| Phase 2: Architecture | 1–2 weeks | Modified TabPFN accepting (time, δ) context labels; K-bin output head |
| Phase 3: Small-scale training | 1 week | SurvPFN trained on 100K synthetic tasks; validation on GBSG/METABRIC |
| Phase 4: Full-scale training + calibration | 2–3 weeks | SurvPFN trained on 1M–3M tasks; hyperparameter sweep |
| Phase 5: Evaluation + baselines | 1–2 weeks | C-index, IBS, D-calibration vs. Cox, RSF, DeepHit, SurvTRACE |
| Phase 6 (optional): Retrieval augmentation | 1–2 weeks | k-NN context selection integrated with SurvPFN inference |
| **Total** | **8–13 weeks** | |

---

## 10. Updated Novel Directions (Post-Survey Ranking)

The two new investigations (Sections 8–9) sharpen the novelty and feasibility picture. Below is a revised ranking of research directions.

### Direction Rankings (Updated)

#### Tier 1: Highest Priority — Clearly Novel, Feasible, High Impact

**1A. SurvPFN: Meta-Training TabPFN for Survival (Core Contribution)**
- **Novelty:** Confirmed novel (no prior work meta-trains on synthetic censored survival data). First in-context learning model for survival analysis.
- **Feasibility:** High (3-month timeline confirmed, see Section 9.4).
- **Impact:** If successful, enables single-forward-pass survival prediction with no fine-tuning — a fundamentally new paradigm for clinical survival analysis, especially in small-N settings.
- **Risk:** Prior mismatch (moderate risk; mitigable).
- **Venue fit:** ICLR 2026, NeurIPS 2026, CHIL 2026.

**1B. Retrieval-Augmented SurvPFN (Natural Extension / Joint Contribution)**
- **Novelty:** Confirmed novel at intersection (TabR + k-NNKM + KATE analogs, none applied to survival ICL). Most directly novel axis: censoring-aware retrieval in an ICL setting.
- **Feasibility:** High — once SurvPFN exists, adding k-NN context selection is ~1–2 weeks of additional work.
- **Impact:** Addresses the N >> 1000 scalability limitation of pure SurvPFN; allows deployment on large cohorts (SEER, TCGA) by retrieving a K-sample context. Directly tests whether retrieved context improves calibration and discrimination over random context.
- **Recommended ablation design:**
  - Baseline: SurvPFN with random K-sample context
  - Variant 1: SurvPFN with k-NN context (Euclidean in feature space)
  - Variant 2: SurvPFN with k-NN context (learned TabPFN embedding space)
  - Variant 3: SurvPFN with label-stratified diverse retrieval
  - Compare all against: Cox, RSF, XGBoost-Cox, DeepSurv, DeepHit, SurvTRACE
- **Risk:** Low — if retrieval doesn't help, it is itself an interesting negative result with clear explanation (ICL already does implicit soft retrieval via attention).

#### Tier 2: Medium Priority — Strong Contributions but More Incremental

**2A. Competing Risks SurvPFN**
- **Novelty:** High — extends 1A to competing risks, which adds both methodological and clinical value.
- **Feasibility:** Medium — requires additional output head engineering; best done after 1A is working.
- **Timeline:** Adds 2–4 weeks on top of 1A. Feasible in 3 months only with 2 engineers.

**2B. Calibration-First Deep Survival Model (Section 7, Direction 4)**
- **Novelty:** Medium — systematic calibration study is valuable but does not require a new architecture.
- **Feasibility:** High — largely empirical.
- **Recommendation:** Package as a secondary contribution / analysis section within the SurvPFN paper rather than a standalone paper.

**2C. Cross-Dataset Tabular FM Pretraining for Survival (Section 7, Direction 3)**
- **Novelty:** Medium — SurvTRACE already does in-domain pretraining; extension to cross-dataset pretraining is incremental.
- **Feasibility:** Medium — data collection bottleneck.
- **Recommendation:** Lower priority given SurvPFN's more fundamental novelty.

#### Tier 3: Lower Priority

**3A. LLM Serialization for Survival (Section 7, Direction 5)**
- Conceptually interesting but hard to control and evaluate rigorously. Best as a future work direction.

**3B. Uncertainty Quantification for SurvPFN**
- SurvPFN's Bayesian-inspired meta-training naturally produces predictive distributions. Conformal prediction wrappers for survival (Candès group work, 2023 [verify]) could be applied post-hoc. Valuable but secondary.

---

### Updated Priority Matrix

| Direction | Novelty (1–5) | Feasibility (1–5) | Impact (1–5) | 3-month feasible? | Recommended? |
|-----------|--------------|-------------------|--------------|-------------------|--------------|
| SurvPFN (meta-train, single risk) | 5 | 4 | 5 | Yes (1 engineer) | **YES — core contribution** |
| Retrieval-aug SurvPFN | 4 | 5 | 4 | Yes (add-on) | **YES — include in paper** |
| Competing risks SurvPFN | 4 | 3 | 4 | Yes (2 engineers) | Yes if resources allow |
| Calibration study | 3 | 5 | 3 | Yes | As secondary analysis |
| Cross-dataset FM pretraining | 3 | 3 | 3 | Partial | Lower priority |
| LLM serialization | 3 | 3 | 2 | Partial | Future work |
| UQ for SurvPFN | 4 | 4 | 4 | Partial | Include if SurvPFN is ready early |

---

### Key New Insights from This Survey

1. **TabR (Gorishniy et al., 2023)** is the most important adjacent work to be aware of and to differentiate from. SurvPFN with retrieval is conceptually related to TabR but: (a) operates in the ICL / meta-learning paradigm (no fine-tuning), (b) handles censored labels, (c) is meta-trained on synthetic data. These are clear differentiators.

2. **k-NN Kaplan-Meier** (Beran 1981, classical) must be included as a baseline in any retrieval-augmented survival paper. It is the canonical non-parametric analog of what SurvPFN with retrieval is doing in the neural setting.

3. **Censoring-aware retrieval** is a genuinely understudied problem. Should the retrieval metric account for censoring (e.g., downweight censored neighbors)? This is an open question. Initial recommendation: retrieve by feature similarity only (censoring-agnostic retrieval), and let the survival model handle censored context labels — consistent with k-NNKM's approach.

4. **Prior calibration** is the highest-risk step in SurvPFN development. Budget at least 1 week for this, with empirical validation against a held-out real dataset (suggested: GBSG as validation, METABRIC/SUPPORT2 as test).

5. **Compute is not the bottleneck.** Training time (24–72 GPU-hours) is manageable on a university cluster or modest cloud budget (~$300–1500). Engineering time is the bottleneck (6–10 weeks for 1 engineer).

6. **Framing recommendation:** The combined SurvPFN + retrieval paper is stronger than either alone. Frame retrieval as addressing the scalability limitation of pure ICL (N > 1000), and show that retrieved context improves over random context on medium-N datasets (N = 500–5000). This directly answers the key empirical question and distinguishes the work from both TabR and k-NNKM.

---

## 11. Session Update — 2026-03-30 (TabDPT & TabICL Embedding Results)

### 11.1 New Contributions Implemented

**TabDPT Frozen Embeddings for Survival Analysis (First Application)**
- Architecture: PFN adapted for EHR/clinical tabular data with flash attention and K-NN retriever
- Embedding extraction: forward hook on `model.head` capturing `src[eval_pos:, 0, :]` — query-token representations after cross-attention with context
- Preliminary result: SUPPORT2 fold-1, `tabdpt_embedding_cox` = 0.576 vs `tabpfn_embedding_cox` = 0.505 (Δ = +0.071)
- **Interpretation**: EHR-domain pretraining provides a non-trivial frozen survival signal, confirming that domain proximity matters even without explicit censoring-aware training.

**TabICL Frozen Embeddings for Survival Analysis (First Application)**
- Architecture: 3-stage pipeline — ColEmbedding (distribution-aware column tokens) → RowInteraction (per-row CLS attention) → ICLearning (dataset-level in-context transformer)
- Two hook points: pre-ICL (post-RowInteraction, label-agnostic) and post-ICL (post-ICLearning blocks[-1], label-conditioned, default)
- Checkpoint: HuggingFace `jingang/TabICL-clf` (v1.1-0506)
- Results: pending (runs in progress)

### 11.2 New Empirical Findings (Updated Results)

**TabPFN frozen embedding (all 4 heads) on FLCHAIN, fold 1:**
- emb-DeepHit: C-index = 0.921 (competitive with best baseline!)
- emb-PCH: C-index = 0.896
- emb-Cox: C-index = 0.821 (improved from previous 0.694 with 2 folds)
- emb-MTLR: C-index = 0.368 ← CATASTROPHIC FAILURE

**Key insight on head sensitivity**: The same frozen TabPFN embedding produces C-index ranging from 0.368 to 0.921 depending on survival head. This reveals that frozen FM representations encode a fragile, order-sensitive signal. The MTLR head's monotone discretization amplifies the mismatch between classification-pretrained representations and the survival time axis, while DeepHit's ranking loss extracts a useful ordering even from weak embeddings.

**Practical rule**: When using frozen FM embeddings, prefer DeepHit or PCHazard heads over MTLR.

### 11.3 N-Dependence Hypothesis — Revised

TabPFN jt-Cox vs standalone DeepHit (consistent comparison across 6 datasets):
- Veterans (N=137): +0.024 ← FM advantage
- WHAS500 (N=500): +0.018 ← FM advantage
- METABRIC (N=1904): +0.003 ← marginal FM advantage
- GBSG (N=2232): +0.009 ← FM advantage
- FLCHAIN (N=6524): -0.020 ← FM DISADVANTAGE
- SUPPORT2 (N=9105): +0.016 ← FM advantage (context subsampling acts as regularizer)

**Revised finding**: TabPFN joint outperforms standalone DeepHit on 5/6 datasets. The sole reversal at FLCHAIN (N=6,524) confirms context-window saturation. SUPPORT2 (N=9,105) is an apparent anomaly — TabPFN still wins because DeepHit struggles at very large N with the current hyperparameter setup, while TabPFN's subsampling regularizes effectively.

### 11.4 Citation Notes

- **TabDPT** (tabdpt2024): cited as preprint/misc in refs.bib. No arXiv ID yet; cite as software/preprint.
- **TabICL** (ye2024tabicl): Ye et al. 2024, ICML 2024 (verify publication venue).
- Both are the first applications to survival analysis — a clear novelty contribution.

---

## 12. Zero-Shot ICL Survival Analysis — Session Update 2026-04-01

### 12.1 Kim, Lai & Zhang (2026) — "Tabular Foundation Models Can Do Survival Analysis"

- **arXiv:** 2601.22259 (submitted January 2026)
- **Key idea:** Reformulate survival analysis as K binary classification tasks at K discrete time boundaries {t_1, …, t_K}. For patient i and bin k, create label Y_{ik} = 𝟙(T_i ≤ t_k), included only when t_k < C_i (censoring mask). The full expanded dataset is fed to a pretrained TFM (TabPFN or MITRA) via in-context learning — no parameter updates.
- **Loss:** Masked binary cross-entropy: L = (1/n) Σ_i Σ_k 𝟙(t_k < C_i) · BCE(p̂(X_i, t_k), Y_{ik}). Right-censored observations contribute only to bins before their censoring time.
- **Theoretical guarantee (Theorem 3.1):** Under conditionally independent censoring, minimising the masked BCE recovers the true survival probabilities P(T > t_k | X) asymptotically. This is the key statistical justification for the surrogate.
- **ICL deployment:** Time t_k is appended as an extra feature, creating rows (X_i ⊕ t_k, Y_{ik}). A single TFM forward pass answers all K queries for a test patient.
- **Dynamic extension:** For longitudinal covariates, H_{i,t_k} (covariate history) and prior survival status are appended per row, enabling landmark-style predictions.
- **Monotonicity:** Post-hoc isotonic regression enforces non-increasing S(t).
- **Datasets:** 43 SurvSet datasets (median N=461) + 5 dynamic datasets.
- **Results (static, mean C-index ± SE):**
  | Model | C-index |
  |---|---|
  | MITRA | 0.677 ± 0.016 |
  | XGBoost | 0.686 ± 0.017 |
  | CoxPH | 0.682 ± 0.017 |
  | DeepHit | 0.574 ± 0.013 |
  MITRA achieves best average rank (2.8) across C-index, IBS, and Integrated AUC. Correlation r=0.89 between BCE loss and IBS validates the classification surrogate.
- **Limitations:** No principled K selection; no ablation on bin granularity; DeepHit/DeepSurv underperformance not addressed.
- **Relation to SurvPFN:** This paper validates the zero-shot ICL approach for survival. Our work extends it to TabPFN v2, TabDPT, and TabICL; compares against frozen-embedding + survival head variants; and evaluates on clinical EHR datasets not covered in their benchmark.

### 12.2 TabSurv (medRxiv 2025.10.03.25337265) — Regression on Uncensored Observations

- **Venue:** medRxiv preprint, October 2025
- **Key idea:** Frame survival prediction as standard regression, **discarding all censored observations** from the ICL context. TabPFN predicts a continuous event time T_i using only uncensored (X_i, T_i) pairs as context. No survival-specific loss; no time-bin discretisation.
- **Key contributions:**
  1. Regression-on-uncensored ICL using TabPFN — zero-shot, no fine-tuning.
  2. Counterfactual treatment arm estimation: duplicates each test instance across candidate treatment arms within a single forward pass.
  3. **Stability score:** Novel metric combining mean C-index with its variability across splits, penalising inconsistent models.
- **Datasets:** 12 breast cancer genomic datasets (RNA-seq/microarray; mix of METABRIC, NKI, UPP, GSE6532, etc.).
- **Results:** TabSurv competitive or superior to all seven baselines (LogisticHazard, PMF, DeepHit, PCHazard, MTLR, DeepSurv, RSF) on C-index and stability score, particularly in high-dimensional low-sample settings.
- **Limitations:** Discarding censored data is statistically inefficient and biased under informative censoring; no competing-risk support; restricted to genomic high-dimensional data; no full survival function output.
- **Contrast with Kim et al.:**
  | | Kim et al. (2026) | TabSurv (2025) |
  |---|---|---|
  | Censoring strategy | Mask BCE at t > C_i | Discard censored cases |
  | Bias | Low (theoretically consistent) | Higher (informative censoring) |
  | Info retention | High | Low (50%+ discarded) |
  | Output | Full S(t) | Point estimate of T |
- **Relation to SurvPFN:** TabSurv is a simple baseline. Kim et al.'s masked BCE approach is more principled and is the basis of our `ZeroShotSurvivalPredictor`. Our contribution: apply the masked BCE framework to TabPFN v2, TabDPT, TabICL, and compare under a consistent 5-fold CV protocol on 6 standard benchmark datasets.

### 12.3 Implementation Notes

Our `ZeroShotSurvivalPredictor` (`survpfn/models/zeroshot_surv.py`) implements the Kim et al. algorithm with two modes:
- `single_context`: time t_k appended as feature; FM fit once on expanded context (faithful to arXiv:2601.22259).
- `per_bin`: FM fit separately per bin without time appended (ablation baseline).

Both modes enforce monotonicity via `sklearn.isotonic.IsotonicRegression`. Benchmark integration adds `tabpfn_zeroshot`, `tabdpt_zeroshot`, `tabicl_zeroshot` (and `*_perbin` variants) to `benchmark.py` and `run.sh`.

---

## 13. Transformer-Based Survival Analysis — Session Update 2026-04-01

### 13.1 SurvTRACE (Wang & Sun, 2022)

- **Paper:** "SurvTRACE: Transformers for Survival Analysis with Competing Events"
- **Authors:** Zifeng Wang, Jimeng Sun (UIUC)
- **Venue:** ACM BCB 2022 (arXiv:2110.00855)
- **Architecture:** BERT-style transformer trained from scratch on tabular survival data. Feature-value pairs are tokenised and embedded; the shared encoder feeds competing-risk-specific output heads. Multi-task auxiliary objectives (masked feature reconstruction) pre-train the encoder before survival fine-tuning.
- **Key novelty:**
  - First transformer applied to *competing-risk* tabular survival analysis.
  - Explicitly models confounders causing selection bias in multi-event observational settings.
  - Attention weights provide covariate-level interpretability.
- **Results:**
  - METABRIC: TD C-index (IPCW) ~0.735 @ 0.25 quantile
  - SUPPORT: TD C-index ~0.669 @ 0.25 quantile
  - SEER (n~470k): claims "all-around superiority" over DeepHit/DeepSurv/RSF
- **Limitation:** Trained from scratch — requires large N; no transferable pre-trained weights; feature schema is fixed to training data.
- **Relevance to SurvPFN:** Direct baseline for METABRIC and SUPPORT2. Use their TD C-index numbers as benchmark targets. Competing-risk architecture is a model for our multi-event Sirbu evaluation.

### 13.2 OSTransformer (Caruso et al., 2024)

- **Paper:** "A Deep Learning Approach for Overall Survival Prediction in Lung Cancer with Missing Values"
- **Authors:** Caruso, Guarrasi, Ramella, Soda (Campus Bio-Medico University of Rome)
- **Venue:** *Computer Methods and Programs in Biomedicine*, 254, 2024 (arXiv:2307.11465)
- **Architecture:** Transformer encoder with named-feature positional encoding. Missing values are handled natively by masking the corresponding token in self-attention — no imputation required. Cause-specific MLP subnets branch from the shared encoder; trained with survival log-likelihood + ranking loss.
- **Key novelty:**
  - First use of masked self-attention as an imputation-free strategy for tabular survival data.
  - Directly addresses high-missingness clinical settings (NSCLC, real-world EHR).
- **Results (NSCLC private dataset, 6-year follow-up):**
  - Ct-index: 71.97 (1-month), 77.58 (1-year), 80.72 (2-year)
  - Outperforms all baselines regardless of imputation method used in comparators.
- **Limitation:** Single private dataset; non-standard Ct-index metric makes cross-paper comparison difficult.
- **Relevance to SurvPFN:** The native-masking idea is directly applicable to the Sirbu dataset (up to 99% missing in LVH/VES). A masked-attention variant could replace our current aggressive `dropna()` strategy.

### 13.3 SA Transformer (Hu et al., 2021)

- **Paper:** "Transformer-Based Deep Survival Analysis"
- **Authors:** Shi Hu, Egill Fridgeirsson, Guido van Wingen, Max Welling (University of Amsterdam)
- **Venue:** AAAI Spring Symposium on Survival Prediction — PMLR Vol. 146, pp. 132–148, 2021
- **Architecture:** Vanilla transformer encoder on patient feature sequences. Ordinal regression models discrete-time survival probabilities over a time grid; a pairwise ranking loss penalises discordant pairs.
- **Key novelty:**
  - First paper to apply self-attention to survival analysis (predates SurvTRACE).
  - Introduces MAE on uncensored subjects as a complementary metric to C-index, arguing that C-index alone is insufficient for model selection.
- **Results:** METABRIC and one additional public dataset; C-index competitive with DeepHit; paper highlights MAE superiority but no single headline C-index is reported.
- **Limitation:** Workshop paper; codebase targets PyTorch 1.1/Python 3.6 (obsolete); single-event only; thin reproducibility.
- **Relevance to SurvPFN:** Historical baseline; consider adopting MAE alongside C-index as a secondary metric in our evaluation.

### 13.4 SAT — Survival Analysis Transformer Toolkit (open-disease-risk, 2024)

- **Repo:** https://github.com/open-disease-risk/sat
- **Authors/Maintainers:** Dominik Dahlem, Mahed Abroshan (GPL-3.0, active 2024–present)
- **No associated paper** — this is a production-grade framework, not a standalone publication.
- **Architecture:** Full pipeline (tokenise → pre-train → fine-tune → inference) built on HuggingFace Transformers (BERT backbone) with Hydra configuration. Composable MetaLoss combines NLLPCHazard, DeepHit, SurvivalFocalLoss, and ranking losses (SampleRanking, MultiEventRanking, ListMLE variants) with five dynamic balancing strategies (fixed, scale, gradient, uncertainty, adaptive). Multi-task heads support simultaneous survival + classification + regression objectives.
- **Key features:**
  - MoCo-enhanced loss buffer for highly censored data.
  - Native MEDS healthcare data format (compatible with FEMR/CLMBR cohorts).
  - Built-in EDA: Weibull/LogNormal/LogLogistic fitting, censoring bias tests.
  - Optuna HPO and k-fold CV with IPCW Brier/C-index logging.
- **Limitation:** No associated paper or published C-index results; empirical validation absent.
- **Relevance to SurvPFN:** The composable loss design (DeepHit + ranking + focal + MoCo buffer) is worth adopting for our survival heads. The MEDS format integration is directly applicable for Sirbu/URRAH EHR data loading.

### 13.5 Summary Comparison

| Model | Year | Architecture | Competing risks | Missing data | Pre-trained | Best C-index reported |
|---|---|---|---|---|---|---|
| SA Transformer | 2021 | Vanilla Transformer + ordinal | ✗ | ✗ | ✗ | ~0.64 METABRIC |
| SurvTRACE | 2022 | BERT from scratch + multi-head | ✓ | ✗ | ✗ | 0.735 METABRIC TD |
| OSTransformer | 2024 | Masked self-attention | ✓ | ✓ (native) | ✗ | 80.72 Ct-index NSCLC |
| SAT toolkit | 2024 | BERT + composable losses | ✓ | ✗ | ✗ | N/A (no paper) |
| **SurvPFN (ours)** | **2026** | **Frozen pre-trained FM** | **✓** | **via FM** | **✓** | **TBD** |

**Key differentiator for SurvPFN:** All four transformer approaches train from scratch on the target dataset. SurvPFN is the first to use *pre-trained* tabular foundation models (TabPFN, TabDPT, TabICL) as frozen encoders, enabling zero-shot or few-shot survival prediction without any architecture-specific training.
