# Breast Cancer Diagnostic Predictor
### End-to-End Machine Learning Project | Clinical Impact Analysis

---

## Project Overview

This project builds a machine learning model to predict whether a breast tumour is **malignant or benign** using real biopsy data from the University of Wisconsin Hospital. The goal is not simply to achieve high accuracy — it is to evaluate the model through a **clinical lens**, where the cost of a missed cancer (false negative) is fundamentally different from the cost of a false alarm (false positive).

> *"At the default threshold, the XGBoost model identifies 92.9% of malignant cases with 97.5% precision — catching 23 additional cancers compared to an unassisted baseline and preventing an estimated $2.99M in avoidable late-stage treatment costs."*

---

## Dataset

**Source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

| Property | Detail |
|---|---|
| Records | 569 patients |
| Features | 30 numeric cell nucleus measurements |
| Target | `diagnosis` — M (Malignant) or B (Benign) |
| Class split | 62.7% Benign / 37.3% Malignant |
| Missing values | None |

Features are computed from digitized images of Fine Needle Aspirate (FNA) biopsies. Each feature is measured as **mean**, **standard error**, and **worst** value across cell nuclei, covering:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave Points, Symmetry, Fractal Dimension

---

## Tools & Technologies

| Phase | Tool |
|---|---|
| Data Cleaning | Python (Google Colab) |
| Exploratory Analysis | Python — pandas, matplotlib, seaborn, scikit-learn |
| Model Training & Evaluation | DataRobot (AutoML) |
| Clinical Impact Analysis | Microsoft Excel |
| Visualization & Dashboard | Tableau |

---

## Project Structure

```
breast-cancer-diagnostic-predictor/
│
├── notebooks/
│   ├── breast_cancer_phase1_cleaning.ipynb       # Data cleaning
│   └── breast_cancer_phase2_EDA.ipynb            # Exploratory analysis
│
├── data/
│   ├── data.csv                                  # Original dataset
│   └── breast_cancer_clean.csv                   # Cleaned dataset
│
├── excel/
│   └── breast_cancer_phase4_clinical_impact.xlsx # Clinical impact workbook
│
├── visualizations/
│   ├── chart1_diagnosis_distribution.png
│   ├── chart2_top3_boxplots.png
│   ├── chart3_correlation_heatmap.png
│   ├── chart4_mean_features_comparison.png
│   ├── chart5_pca_scatter.png
│   ├── chart6_feature_importance_correlation.png
│   ├── chart7_feature_impact_datarobot.png
│   ├── chart8_roc_confusion_matrix.png
│   ├── chart9_roc_holdout_confusion_matrix.png
│   └── chart10_metric_scores.png
│
└── README.md
```

---

## Phase 1 — Data Cleaning

**Notebook:** `breast_cancer_phase1_cleaning.ipynb`

Key steps:
- Dropped `Unnamed: 32` — empty ghost column from CSV trailing comma artifact
- Dropped `id` — patient identifier with no predictive value
- Encoded `diagnosis`: M → 1 (Malignant), B → 0 (Benign)
- Verified zero missing values across all 30 features
- Verified zero duplicate patient records
- Confirmed `area_worst` and `perimeter_worst` outliers are clinically valid — malignant tumours genuinely produce extreme cell measurements

**Output:** `breast_cancer_clean.csv` — 569 rows × 31 columns, zero nulls, zero duplicates

---

## Phase 2 — Exploratory Data Analysis

**Notebook:** `breast_cancer_phase2_EDA.ipynb`

### Key Findings

**1. Class Imbalance**
- 357 Benign (62.7%) vs 212 Malignant (37.3%)
- A model predicting "Benign" for every patient achieves 62.7% accuracy while missing 100% of cancers
- This is why **Recall (Sensitivity)** — not accuracy — is the primary evaluation metric

**2. Top Predictors (Manual Correlation Analysis)**

| Rank | Feature | Correlation with Malignancy |
|---|---|---|
| 1 | concave points_worst | 0.794 |
| 2 | perimeter_worst | 0.783 |
| 3 | concave points_mean | 0.777 |
| 4 | radius_worst | 0.776 |
| 5 | perimeter_mean | 0.742 |

All top predictors measure **cell size** or **shape irregularity** — consistent with how pathologists diagnose cancer manually under a microscope.

**3. PCA Separation**
Even when compressed to just 2 dimensions (capturing 63.3% of variance), Benign and Malignant cases form visually distinct clusters — confirming the features carry strong discriminative signal.

**4. Core Clinical Narrative**
> *"Malignant tumours are characterized by larger cell size and greater shape irregularity. These two biological properties drive the model's predictive power and are consistent with pathological diagnostic criteria."*

---

## Phase 3 — Model Training (DataRobot)

**Platform:** DataRobot AutoML  
**Configuration:**
- Learning type: Supervised
- Target: `diagnosis` (Binary Classification)
- Positive class: 1 (Malignant)
- Optimization metric: **AUC** (not accuracy — clinically appropriate for imbalanced classes)
- Validation: Training-validation-holdout (64% / 16% / 20%)
- Sampling: Stratified

### Model Leaderboard (Top Models by Holdout AUC)

| Model | Holdout AUC |
|---|---|
| eXtreme Gradient Boosted Trees | **0.9855** |
| Keras Neural Network (64%) | 0.9868 |
| Random Forest | 0.9818 |
| Light Gradient Boosted Trees | 0.9818 |

### Selected Model: eXtreme Gradient Boosted Trees (XGBoost)

**Why XGBoost over the Keras Neural Network:**
- More consistent generalization to unseen data
- Produces interpretable feature importance scores
- In clinical settings, explainability matters — a model must communicate *why* it flagged a patient, not just *that* it did

### Model Performance (Holdout Set — 114 Unseen Patients)

| Metric | Score |
|---|---|
| AUC | **0.9855** |
| F1 Score | **0.9512** |
| Sensitivity / Recall | **0.9286** |
| Precision | **0.9750** |
| Gini Norm | 0.9709 |
| Max MCC | 0.9245 |
| Rate @ Top 10% | **1.0000** |

### Confusion Matrix (Holdout)

| | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 71 ✅ TN | 1 ⚠️ FP |
| **Actual Malignant** | 3 ❌ FN | 39 ✅ TP |

**Rate@Top10% = 1.000** — Every malignant case appears in the top 10% highest-risk scores. A hospital screening the top 10% of flagged patients would catch every cancer in this dataset.

### Feature Importance (SHAP Values)

| Rank | Feature | Relative Impact |
|---|---|---|
| 1 | concave points_worst | 100% |
| 2 | smoothness_worst | 95% |
| 3 | perimeter_worst | 88% |
| 4 | radius_worst | 85% |
| 5 | area_se | 78% |
| 6 | texture_mean | 72% |

**Notable:** The model independently confirmed 3 of the top 5 features identified manually in Phase 2 EDA — validating the exploratory analysis approach.

### The False Negative Argument
> *"At the default threshold, 3 malignant cases out of 42 were missed — a false negative rate of 7.1%. In a clinical decision support tool, this threshold should be adjusted downward to prioritize catching every possible malignancy, accepting slightly more false alarms. This threshold decision belongs to clinicians, not algorithms."*

---

## Phase 4 — Clinical Impact Analysis (Excel)

**File:** `breast_cancer_phase4_clinical_impact.xlsx`

### Workbook Structure
1. **Model Summary** — All performance metrics in one reference table
2. **Confusion Matrix** — Color-coded breakdown of prediction outcomes
3. **Clinical Cost Analysis** — Human and financial cost of diagnostic errors
4. **Baseline Comparison** — Model vs no-model vs perfect model
5. **Dashboard Summary** — Clean summary tables for Tableau

### Key Findings

**Human Impact:**
- 5-year survival rate for early-stage breast cancer: **99%**
- 5-year survival rate for late-stage breast cancer: **27%**
- Each missed malignancy risks a **72% reduction in survival probability** if detected late

**Financial Impact:**

| Scenario | Cases Caught | Cases Missed | Treatment Cost |
|---|---|---|---|
| No Model (Baseline) | 16 | 26 | $3,380,000 |
| **XGBoost Model** | **39** | **3** | **$390,000** |
| Perfect Model | 42 | 0 | $0 |

- **23 additional cancers caught** compared to unassisted baseline
- **$2,990,000 in avoidable treatment costs prevented** on holdout set alone
- Scaled to full dataset: approximately **$1.95M in preventable late-stage treatment costs**

---

## Phase 5 — Tableau Visualizations

Three analytical visualizations built from the cleaned patient-level dataset:

**1. Scatter Plot — Cell Irregularity vs Perimeter**
- Each dot = one patient, colored by diagnosis
- Separate trend lines per class with confidence bands
- Visually proves the natural separation between Benign and Malignant cases

**2. Box Plot Dashboard — Top 6 Feature Distributions**
- Side-by-side comparison of Benign vs Malignant for 6 strongest predictors
- Confirms Malignant cases consistently show higher values and wider spread

**3. Heatmap — Average Feature Values by Diagnosis**
- All 30 features normalized to 0-1 scale
- Color intensity shows discriminative power at a glance
- Strongest contrast: Area Worst (558 Benign vs 1,422 Malignant — 2.5x difference)

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | eXtreme Gradient Boosted Trees |
| Holdout AUC | 0.9855 |
| Cancers Correctly Identified | 39 / 42 (92.9%) |
| False Negative Rate | 7.1% |
| Additional Cases vs Baseline | +23 cancers caught |
| Preventable Treatment Cost Savings | $2,990,000 |

---

## Key Takeaways

1. **Accuracy is the wrong metric for medical diagnosis.** A model predicting "Benign" for every patient scores 62.7% accuracy while missing every cancer. Recall and AUC are the correct metrics.

2. **Cell shape irregularity and size are the dominant malignancy markers.** `concave points_worst` and `perimeter_worst` consistently rank as the top predictors — consistent with pathological diagnostic criteria.

3. **Threshold matters more than model choice.** The difference between catching 39 and 42 malignant cases is not about the algorithm — it is about where you set the classification threshold. Lower thresholds catch more cancers at the cost of more false alarms.

4. **Clinical impact must be quantified in human and financial terms.** A model with 0.9855 AUC is meaningless to a hospital administrator. "$2.99M in preventable treatment costs" and "23 additional cancers caught" are meaningful.

5. **Explainability is non-negotiable in healthcare.** XGBoost was chosen over a marginally better neural network because feature importance scores allow clinicians to understand and trust the model's reasoning.

---

## About

**Author:** Muzammil Ansari  
**Program:** Post-Baccalaureate Diploma in Business Analytics, University of the Fraser Valley  
**LinkedIn:** [linkedin.com/in/muzammilansari7494](https://linkedin.com/in/muzammilansari7494)  
**Location:** Abbotsford, BC, Canada

---

*Data source: Breast Cancer Wisconsin (Diagnostic) Dataset — UCI Machine Learning Repository. This project is for educational and portfolio purposes. The model is not intended for clinical use.*
