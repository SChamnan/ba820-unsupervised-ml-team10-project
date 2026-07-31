# BA820 — Unsupervised ML | Team 10 Project

**European Drugs Development:** discovering structure in EU medicine authorization data using unsupervised learning.

This repo stores and documents each milestone (M1 → M4) of the BA820 Unsupervised Machine Learning group project.

**Team 10:** Chamnan Suon · Gurveen Rekhi · Roberto Albornoz · Samuel Buelvas

---

## Project Overview

Drug development in Europe shapes the pharmaceutical industry worldwide. Using the European Medicines Agency (EMA) medicine registry, this project applies clustering, association-rule mining, and NLP/text-embedding methods to answer: **do medicines fall into natural regulatory and therapeutic "types," and do those types relate to authorization outcomes?**

Two candidate datasets were explored in the early phase; **European Drugs Development** was selected as the primary dataset and **Childcare Costs** was retained as exploratory/backup work.

### Domain Questions

| # | Question | Primary methods |
|---|----------|-----------------|
| 1 | **Regulatory paths** — do special approval pathways co-occur in consistent combinations? | Hierarchical clustering, K-Means, elbow/silhouette |
| 2 | **Patterns in approvals** — are there natural clusters of authorized drugs, and are they stable over time? | Hierarchical, K-Means (k=4, k=6), ATC-code contrast |
| 3 | **Patterns in rejections** — which regulatory features associate with refusal? | Association rules (Apriori), hierarchical, K-Means on TF-IDF |
| 4 | **Veterinary medicines** — how do veterinary drugs differ structurally from human drugs? | Species segmentation, ATC label analysis, MBA, clustering |

---

## Repository Structure

```
ba820-unsupervised-ml-team10-project/
├── european-drugs-development/          # PRIMARY dataset + M1 EDA
│   ├── drugs-dataset.csv                # 1,988 medicines × 28 columns (EMA)
│   ├── european_drugs_development.ipynb # M1: EDA & feasibility
│   └── README.md
│
├── childcare-costs/                     # Secondary / exploratory dataset
│   ├── childcare_costs.csv              # 34,567 county-year rows × 61 cols
│   ├── counties.csv                     # 3,144 county reference rows
│   └── Childcare_Cost_EDA.ipynb
│
├── Project-M2/                          # Milestone 2 — individual notebooks
│   ├── Chamnan_Suon_M2_...ipynb
│   ├── Gurveen_Rekhi_M2_...ipynb
│   ├── Roberto_Albornoz_M2_...ipynb
│   └── Samuel_Buelvas_...ipynb
│
├── Project-M3/
│   └── M3_European_Drugs_Development.ipynb   # Consolidated team notebook
│
├── Project-M4/                          # Milestone 4 — individual deep dives
│   ├── Chamnan_Suon_M4_...ipynb         # Assoc. rules + text embeddings
│   ├── Gurveen_Rekhi_M4_...ipynb        # MBA for anomaly detection
│   ├── Roberto_Albornoz_M4_...ipynb     # Market basket analysis
│   └── Samuel-Buelvas-M4-...ipynb       # BoW / TF-IDF / GloVe / SBERT + SVD, PCA
│
├── Chamnan_Suon_M4_European_Drugs_Development.ipynb   # Root-level copy of M4 notebook
└── README.md
```

---

## Milestone Progression

| Milestone | Focus | What was added |
|-----------|-------|----------------|
| **M1** | Dataset selection & EDA | Authorization trends over time, top MAH companies, top therapeutic areas, special-approval overlap, human vs. veterinary comparison, text-variability check for NLP feasibility |
| **M2** | First unsupervised pass (individual) | Shared cleaning pipeline; hierarchical clustering & dendrograms, K-Means, K-Modes, early Apriori rules, TF-IDF on therapeutic text |
| **M3** | Consolidated team analysis | All four domain questions in one notebook; silhouette/elbow validation, χ² tests, `MultiLabelBinarizer` on ATC codes, association rules, species segmentation, joint Key Findings + Next Steps |
| **M4** | Individual specialization | Market basket analysis with threshold calibration & sensitivity/robustness testing; outcome-linked rules (authorised / withdrawn / refused); text pipeline comparison (BoW → TF-IDF → GloVe → SBERT); dimensionality reduction via Truncated SVD and PCA; latent semantic cluster contrast |

---

## Primary Dataset — `drugs-dataset.csv`

EMA medicine registry, **1,988 records × 28 fields**. Key columns:

- **Identity:** `medicine_name`, `common_name`, `active_substance`, `product_number`, `category` (human / veterinary), `species`
- **Classification:** `therapeutic_area`, `atc_code`, `pharmacotherapeutic_group`, `condition_indication`
- **Regulatory pathway flags (boolean):** `generic`, `biosimilar`, `conditional_approval`, `exceptional_circumstances`, `accelerated_assessment`, `orphan_medicine`, `additional_monitoring`, `patient_safety`
- **Outcome & timing:** `authorisation_status`, `marketing_authorisation_date`, `date_of_refusal_of_marketing_authorisation`, `date_of_opinion`, `decision_date`, `revision_number`, `revision_date`
- **Other:** `marketing_authorisation_holder_company_name`, `first_published`, `url`

---

## Methods & Stack

**Language:** Python (Jupyter / Google Colab — notebooks load data directly from GitHub raw URLs)

| Category | Tools |
|----------|-------|
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `yellowbrick` (`KElbowVisualizer`, `SilhouetteVisualizer`) |
| Clustering | `sklearn` (`KMeans`, `AgglomerativeClustering`), `scipy` (`linkage`, `dendrogram`, `pdist`), `kmodes` |
| Cluster validation | `silhouette_score`, `kneed.KneeLocator`, elbow method |
| Association rules | `mlxtend` (`TransactionEncoder`, `apriori`, `association_rules`) |
| Text / NLP | `TfidfVectorizer`, `CountVectorizer`, `gensim` (GloVe), `sentence-transformers` (SBERT) |
| Dimensionality reduction | `PCA`, `TruncatedSVD` |
| Statistics | `scipy.stats.chi2_contingency`, `statsmodels` |

**Shared preprocessing helpers** (defined in M1, reused across all milestones): `clean_string_series()` for whitespace/type normalization, plus boolean casting, date parsing, and derived feature engineering.

---

## Key Findings

- **Regulatory pathways:** Most medicines follow a standard pathway, but special pathways co-occur in consistent, recognizable combinations.
- **Approvals:** Clusters persist over time, yet approval behavior is driven by more than regulatory complexity and therapeutic area alone — clustering on its own does not explain approval rates.
- **Refusals:** `orphan_medicine` and `additional_monitoring` show the strongest association with refusal; `generic` and `biosimilar` pathways are linked to *lower* refusal risk.
- **Text clusters:** Terms recur across multiple clusters, indicating overlapping therapeutic focus rather than cleanly separable categories.
- **Veterinary vs. human:** Several special-approval indicators are rare or absent for veterinary medicines, limiting direct comparison; veterinary drugs cluster independently under ATC classification.

### Known Limitations

Medicines exhibit **continuous, overlapping profiles** rather than sharply separable clusters. Sparse, unevenly distributed binary flags let common pathway combinations dominate centroid- and mode-based methods. Refusals are rare, so detecting associations requires relaxed support thresholds — trading stability for discovery. Broad ATC codes and therapeutic areas further limit clinical granularity.

### Next Steps

Standardize preprocessing and a shared feature dictionary across human and veterinary subsets; apply text embeddings to `therapeutic_area` and `condition_indication`; use PCA primarily for dimensionality reduction and visualization to reduce noise from sparse binary indicators; explore text-to-text regression for outcome linkage.

---

## Getting Started

```bash
git clone https://github.com/SChamnan/ba820-unsupervised-ml-team10-project.git
cd ba820-unsupervised-ml-team10-project

pip install pandas numpy matplotlib seaborn scikit-learn scipy \
            mlxtend yellowbrick kneed kmodes gensim sentence-transformers statsmodels
```

Open any notebook in Jupyter or Google Colab. Notebooks pull the dataset from GitHub raw URLs, so they run without local path configuration.

**Suggested reading order:** `european-drugs-development/european_drugs_development.ipynb` (M1) → `Project-M3/M3_European_Drugs_Development.ipynb` (consolidated analysis) → any `Project-M4/` notebook (individual deep dives).
