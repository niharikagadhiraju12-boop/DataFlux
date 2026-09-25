# Amazon ML Challenge 2026 — Business Entity Resolution

## Project Overview
This repository contains the collaborative solution for the **Amazon ML Challenge 2026: Business Entity Resolution**. The goal of the project is to accurately identify and link records referring to the same real-world business entity across different heterogeneous data sources (Source 1, Source 2, and Source 3).

The primary evaluation metric for this competition is **F0.5**, prioritizing high precision to minimize erroneous entity linkages while maintaining solid recall.

---

## Team Division & Collaboration Structure
All four team members contribute directly to **ONE shared, unified end-to-end pipeline**. The responsibilities are allocated across the project lifecycle as follows:

| Team Member | Core Focus Areas | Key Deliverables & Modules |
| :--- | :--- | :--- |
| **Member 1** | **Data Analysis & Preprocessing** | • `notebooks/01_data_analysis.ipynb`<br>• `notebooks/02_preprocessing.ipynb`<br>• `code/business_entity_resolution/src/preprocessing.py` |
| **Member 2** | **Candidate Generation & Blocking** | • `notebooks/03_blocking.ipynb`<br>• `code/business_entity_resolution/src/blocking.py` |
| **Member 3** | **Feature Engineering & ML Matching Model** | • `notebooks/04_model.ipynb`<br>• `code/business_entity_resolution/src/features.py`<br>• `code/business_entity_resolution/src/model.py` |
| **Member 4** | **Evaluation, Output & Submission Quality** | • `notebooks/05_evaluation.ipynb`<br>• `code/business_entity_resolution/src/evaluation.py`<br>• `code/business_entity_resolution/src/generate_output.py`<br>• `Documentation_template.md`<br>• Final submission validation |

---

## Repository Structure

```
ml/
│
├── code/
│   └── business_entity_resolution/
│       │
│       ├── src/
│       │   ├── preprocessing.py       # Data loading, validation, and text/address cleaning
│       │   ├── blocking.py            # Candidate pair generation and blocking logic
│       │   ├── features.py            # Pairwise feature extraction and similarity metrics
│       │   ├── model.py               # ML matching model definition, training, inference
│       │   ├── evaluation.py          # Metric computation (F0.5, Precision, Recall, etc.)
│       │   └── generate_output.py     # Output generation (matching_results.tsv, candidate_pairs.tsv)
│       │
│       ├── README.md                  # Detailed component documentation
│       └── requirements.txt           # Minimal runtime dependencies
│
├── notebooks/
│   ├── 01_data_analysis.ipynb         # Exploratory data analysis (Member 1)
│   ├── 02_preprocessing.ipynb         # Preprocessing validation & experiments (Member 1)
│   ├── 03_blocking.ipynb              # Blocking & candidate generation experiments (Member 2)
│   ├── 04_model.ipynb                 # Feature engineering & model training (Member 3)
│   └── 05_evaluation.ipynb            # Pipeline evaluation & error analysis (Member 4)
│
├── output/                            # Target directory for generated submission TSVs
│
├── Documentation_template.md          # Comprehensive challenge report documentation template
│
└── README.md                          # Repository root documentation (Team overview)
```

---

## Strict Competition Rules & Guidelines
- **Zero External Data**: Under no circumstances will external business registries, geocoding APIs, public websites, or external databases be used.
- **Data Integrity**: The raw challenge datasets must never be modified or committed into version control.
- **Unified Pipeline**: All individual modules integrate into a reproducible pipeline complying with the official submission specifications.
