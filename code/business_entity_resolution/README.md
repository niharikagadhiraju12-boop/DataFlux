# Business Entity Resolution

## Overview
This component implements the Business Entity Resolution pipeline for the Amazon ML Challenge 2026. The objective is to resolve and match records corresponding to identical business entities across heterogeneous data sources.

## Strict Data & Compliance Rules
- **Challenge Data Only**: The pipeline uses strictly the data supplied by the challenge organizers.
- **No External Data or APIs**: No external business directories, geocoding APIs, entity-resolution APIs, government registries, or synthetic/augmented data are permitted or utilized.
- **Challenge Compliance**: All aspects of data ingestion, processing, modeling, and output formatting must strictly comply with the official challenge README and problem statement.

## Planned Pipeline Architecture
The solution follows a modular multi-stage architecture:

```
Data Loading
     │
     ▼
Preprocessing & Normalization
     │
     ▼
Candidate Generation / Blocking
     │
     ▼
Feature Engineering
     │
     ▼
Matching Model (ML Classifier)
     │
     ▼
Evaluation (F0.5 Score)
     │
     ▼
Output Generation (matching_results.tsv & candidate_pairs.tsv)
```

### Module Responsibilities:
1. **Data Loading & Preprocessing (`src/preprocessing.py`)**:
   - Ingests official challenge TSV files without modifying source data.
   - Cleans and normalizes business names, addresses, and related textual attributes.
2. **Blocking & Candidate Generation (`src/blocking.py`)**:
   - Reduces the quadratic comparison space between sources (Source 1 ↔ Source 2 and Source 1 ↔ Source 3).
   - Generates high-recall candidate pairs for scoring.
3. **Feature Engineering (`src/features.py`)**:
   - Computes pairwise similarity metrics (name, address, token, character, and country agreement).
4. **Matching Model (`src/model.py`)**:
   - Trains and executes the ML classification model to predict entity equivalence.
5. **Evaluation (`src/evaluation.py`)**:
   - Computes the primary competition metric: **F0.5 Score** (placing higher emphasis on precision to minimize false entity matches), alongside Pair Completeness and Reduction Ratio.
6. **Output Generation (`src/generate_output.py`)**:
   - Generates the required TSV deliverables: `matching_results.tsv` and `candidate_pairs.tsv`.
