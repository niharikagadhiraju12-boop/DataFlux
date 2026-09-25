# Amazon ML Challenge 2026 — Documentation Report

## 1. Problem Understanding
<!-- Describe the business entity resolution problem, objectives, and the significance of high-precision record linkage across heterogeneous sources. -->
- **Task Description**: 
- **Challenge Objective**: 
- **Key Challenges in Business Entity Resolution**: 

## 2. Data
<!-- Overview of the challenge-provided dataset. Details will be populated after inspecting the official challenge files. -->
- **Source Datasets**:
  - Source 1:
  - Source 2:
  - Source 3:
- **Attribute Overview**: 
- **Data Quality Observations**: 

## 3. Preprocessing
<!-- Document text cleaning, normalization, handling of missing values, and standardization routines. -->
- **Text & Business Name Normalization**: 
- **Address Standardization**: 
- **Missing Value Handling**: 

## 4. Candidate Generation / Blocking
<!-- Detail the blocking strategies used to prune Cartesian product pairs while maintaining high pair completeness. -->
- **Blocking Key Design**: 
  - Source 1 ↔ Source 2:
  - Source 1 ↔ Source 3:
- **Pair Completeness vs. Reduction Ratio**: 
- **Filtering Thresholds**: 

## 5. Feature Engineering
<!-- Detail the pairwise similarity features extracted for candidate pairs. -->
- **Business Name Similarities**: 
- **Address & Locality Similarities**: 
- **Token & Character Level Metrics**: 
- **Exact / Categorical Agreement Features**: 

## 6. Matching Model
<!-- Explain model selection, architecture, loss function, and threshold tuning. -->
- **Model Architecture**: 
- **Hyperparameter Configuration**: 
- **Training Strategy & Validation Scheme**: 
- **Decision Threshold Optimization (for F0.5)**: 

## 7. Evaluation
<!-- Describe evaluation methodology, official metric formulation, and validation performance. -->
- **Primary Competition Metric**: F0.5 Score
  $$\text{F0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Validation Strategy**: 
- **Metric Tracking (Precision, Recall, F0.5)**: 

## 8. Results
<!-- Record official validation metrics, ablation studies, and performance tables once experiments are run. (No invented values) -->
- **Validation Set Metrics**:
  - Precision: *(To be populated after model training)*
  - Recall: *(To be populated after model training)*
  - F0.5 Score: *(To be populated after model training)*
- **Ablation Studies / Iterations**: 

## 9. Limitations
<!-- Discuss observed edge cases, false positive drivers, noise sensitivity, and constraints. -->
- **Identified Failure Modes**: 
- **Computational / Scalability Constraints**: 
- **Edge Cases in Unstructured Addresses / Names**: 

## 10. Reproducibility
<!-- Provide exact instructions to reproduce the end-to-end pipeline and generate submissions. -->
- **Environment Setup**:
  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  ```
- **Execution Workflow**:
  1. Preprocessing & Data Loading:
  2. Candidate Generation / Blocking:
  3. Feature Extraction:
  4. Model Inference / Prediction:
  5. Submission Generation (`output/matching_results.tsv`, `output/candidate_pairs.tsv`):
