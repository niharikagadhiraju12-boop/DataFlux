# Amazon ML Challenge 2026 — Documentation Report

## 1. Problem Understanding
- **Task Description**: Link and deduplicate business entities across heterogeneous databases (Source 1, Source 2, and Source 3) under zero-external-data rules.
- **Challenge Objective**: Accurately resolve distinct records referring to the same real-world business entity, maximizing the official competition metric ($F_{0.5}$) where precision is weighted 4x over recall.
- **Key Challenges in Business Entity Resolution**: Heavy multilingual variations (Hindi/Devanagari, Tamil, French, Spanish, English), non-standard address formats, corporate suffix differences (Inc., LLC, Pvt Ltd), and vast Cartesian comparison spaces ($O(N 	imes M)$) requiring high-recall candidate blocking.

## 2. Data
- **Source Datasets**:
  - Source 1: Primary query entities (`train_source1.tsv`: 2,206,821 records; `test_source1.tsv`: 1,732,544 records).
  - Source 2: Secondary target entity repository (`train_source2.tsv`: 5,034,616 records; `test_source2.tsv`: 5,285,603 records).
  - Source 3: Tertiary target entity repository (`train_source3.tsv`: 5,285,603 records; `test_source3.tsv`: 5,285,603 records).
- **Attribute Overview**: `entity_id`, `business_name`, `business_address`, `country`.
- **Data Quality Observations**: Missing addresses and countries in subsets, transliteration variations, abbreviations, punctuation noise.

## 3. Preprocessing
- **Text & Business Name Normalization**: Unicode NFKC normalization, Unicode-aware lowercasing (`.casefold()`), safe regex punctuation stripping (`re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)`) preserving multilingual character sets, whitespace standardization.
- **Address Standardization**: Lowercasing, Unicode normalization, whitespace collapsing, tokenization.
- **Missing Value Handling**: NaN conversion to empty strings to ensure deterministic string comparisons.

## 4. Candidate Generation / Blocking
- **Blocking Key Design**:
  - Multi-pass blocking system implemented in `src/blocking.py`:
    1. Country partitioning (strict geographic isolation, preserving open world sets).
    2. Rare / informative name token blocking (filtering high-frequency corporate stopwords).
    3. Character n-gram TF-IDF retrieval (3-4 character n-grams, $k=25$, min similarity 0.18).
    4. Address token blocking (house numbers, postal codes, locality terms).
- **Pair Completeness vs. Reduction Ratio**:
  - Source 2 Candidate Recall: **98.15%** (1,644 / 1,675 true links retained)
  - Source 3 Candidate Recall: **98.94%** (1,775 / 1,794 true links retained)
  - Overall Candidate Recall: **98.56%** (3,419 / 3,469 true links retained)
  - Candidates generated on validation set: 81,756 candidate pairs from 1,000 S1 records (~81.8 candidates/entity).

## 5. Feature Engineering
- **Business Name Similarities**: Exact normalized match, whitespace token Jaccard similarity, character-level edit similarity (`SequenceMatcher.ratio`), absolute character length difference.
- **Address & Locality Similarities**: Exact normalized address match, address token Jaccard similarity, character-level address edit similarity, absolute address length difference.
- **Token & Character Level Metrics**: Name token count difference, address token count difference.
- **Exact / Categorical Agreement Features**: Country exact match (`1` if both non-empty and matching, `0` otherwise).
- **Feature Matrix**: 11 numerical pairwise similarity features extracted via `src/features.py` with shared Unicode preprocessing.

## 6. Matching Model
- **Model Architecture**: Binary Logistic Regression with balanced class weighting (`EntityMatchingModel` in `src/model.py`).
- **Hyperparameter Configuration**: `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.
- **Training Strategy & Validation Scheme**:
  - **Training Split**: First 1,000 Source 1 entities (`iloc[0:1000]`), generating 85,144 candidate pairs.
  - **Validation Split**: Next 1,000 Source 1 entities (`iloc[1000:2000]`), generating 81,756 candidate pairs.
  - **Zero Leakage**: Source 1 training and validation IDs are strictly disjoint (0 overlap). Model fitted exclusively on training candidate pairs; probabilities generated exclusively on held-out validation candidate pairs.
- **Decision Threshold Optimization (for F0.5)**: Decision boundary swept from 0.50 to 0.99 with step 0.01. Optimal threshold identified at **0.97** to suppress balanced weighting false positives.

## 7. Evaluation
- **Primary Competition Metric**: F0.5 Score
  $$\text{F0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Validation Strategy**: 1,000 held-out Source 1 validation entities evaluated against both Source 2 and Source 3 target pools preserving 100% of true ground-truth targets + 20,000 negative distractors.
- **Metric Tracking (Precision, Recall, F0.5)**:
  - Threshold 0.50: Precision = 83.53%, Recall = 97.78%, F0.5 = 86.04% (4,002 predicted matches; 659 FP)
  - Threshold 0.80: Precision = 94.37%, Recall = 95.61%, F0.5 = 94.62% (3,464 predicted matches; 195 FP)
  - Threshold 0.90: Precision = 97.10%, Recall = 94.18%, F0.5 = 96.51% (3,316 predicted matches; 96 FP)
  - Threshold 0.97 (Optimal): Precision = 98.90%, Recall = 92.07%, **F0.5 = 97.46%** (3,183 predicted matches; 35 FP)

## 8. Results
- **Validation Set Metrics (Iteration 1 Measured)**:
  - Precision: **98.90%** (0.9890)
  - Recall: **92.07%** (0.9207)
  - F0.5 Score: **97.46%** (0.9746)
  - Predicted Matches: **3,183** (TP: 3,148, FP: 35, FN: 271)
  - Optimal Decision Threshold: **0.97**
  - Validation S2 Candidate Recall: **98.15%**
  - Validation S3 Candidate Recall: **98.94%**
  - Validation Overall Candidate Recall: **98.56%**
- **Ablation Studies / Iterations**:
  - *Iteration 0 (Member 3 Smoke Test)*: Single 1,000-entity set, in-sample training/evaluation leakage, S2-only truncation (`head(100k)`), uncalibrated threshold 0.50 ($F_{0.5} \approx 6.7\%$).
  - *Iteration 1 (Member 4 Unified Pipeline)*: Disjoint 1k train / 1k val split, multi-source blocking (S2 & S3), Unicode preprocessing integration, out-of-sample inference, F0.5 threshold calibration to 0.97 ($F_{0.5} = 97.46\%$).

## 9. Limitations
- **Identified Failure Modes**:
  - High-confidence False Positives (35 pairs): Co-located distinct businesses sharing identical building complexes/addresses with generic corporate tokens.
  - False Negatives (271 pairs): Heavy cross-script transliteration (e.g., Latin name vs Tamil script `ரெட் பவர் எல்எல்பி`) where string similarity is low and addresses are missing.
- **Computational / Scalability Constraints**: `SequenceMatcher` in pure Python scales quadratically with string length; future iterations should adopt C-accelerated string metrics (`rapidfuzz`).
- **Edge Cases in Unstructured Addresses / Names**: Missing locality/country attributes preventing geographic blocking.

## 10. Reproducibility
- **Environment Setup**:
  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  ```
- **Execution Workflow**:
  1. Preprocessing & Data Loading: `code/business_entity_resolution/src/preprocessing.py`
  2. Candidate Generation / Blocking: `code/business_entity_resolution/src/blocking.py`
  3. Feature Extraction: `code/business_entity_resolution/src/features.py`
  4. Model Inference / Prediction: `code/business_entity_resolution/src/model.py`
  5. Evaluation & Validation: `notebooks/05_evaluation.ipynb`
  6. Submission Generation & Verification:
     - `output/matching_results.tsv` (1,732,544 rows)
     - `output/candidate_pairs.tsv` (1,732,544 rows)
     - Validated via `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` -> **PASS (Exit code 0)**.
