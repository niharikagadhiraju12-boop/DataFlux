import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------
# Setup paths and environment
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(SRC_DIR))

from src.preprocessing import normalize_text, normalize_country
from src.blocking import generate_candidates, export_candidate_pairs_tsv
from src.features import extract_pair_features
from src.model import EntityMatchingModel
from src.evaluation import (
    compute_f_beta_score,
    evaluate_predictions,
    evaluate_threshold_sweep,
    find_best_threshold,
    evaluate_candidate_recall,
)
from src.generate_output import export_matching_results_tsv

DATA_ROOT = PROJECT_ROOT.parent / "6ab10eb3b23ba_student_resource" / "student_resource"
if not DATA_ROOT.exists():
    DATA_ROOT = Path(r"C:\Users\sathw\OneDrive\Desktop\ml\6ab10eb3b23ba_student_resource\student_resource")

TRAIN_DIR = DATA_ROOT / "dataset" / "train"
TEST_DIR = DATA_ROOT / "dataset" / "test"
VALIDATOR_SCRIPT = DATA_ROOT / "utils" / "validate_submission.py"

print("=" * 70)
print("   DATAFLUX ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE    ")
print("=" * 70)

# --------------------------------------------------------------------
# STAGE 1: LOADING DATA
# --------------------------------------------------------------------
print("\n[1/8] Loading data...")
t0_stage1 = time.time()
s1_raw_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", nrows=2000)
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")

print(f"  Loaded {len(s1_raw_df):,} Source 1 rows (first 2,000 for holdout validation).")
print(f"  Loaded {len(gt_df):,} ground truth rows.")
print(f"  Stage 1 completed in {time.time() - t0_stage1:.2f}s")

# --------------------------------------------------------------------
# STAGE 2: CREATING TRAIN/VALIDATION SPLIT
# --------------------------------------------------------------------
print("\n[2/8] Creating train/validation split...")
t0_stage2 = time.time()
train_s1 = s1_raw_df.iloc[0:1000].copy()
val_s1 = s1_raw_df.iloc[1000:2000].copy()

train_s1_ids = set(train_s1["entity_id"])
val_s1_ids = set(val_s1["entity_id"])
overlap = train_s1_ids.intersection(val_s1_ids)

print(f"  Train Source 1 count:      {len(train_s1_ids):,}")
print(f"  Validation Source 1 count: {len(val_s1_ids):,}")
print(f"  ID overlap:                {len(overlap)}")
assert len(overlap) == 0, f"Error: Source 1 train and validation IDs overlap ({len(overlap)})! Must be 0."

def parse_ground_truth(gt, s1_ids):
    sub = gt[gt["source1_entity_id"].isin(s1_ids)]
    all_pairs = set()
    s2_pairs, s3_pairs = set(), set()
    needed_s2, needed_s3 = set(), set()
    for _, r in sub.iterrows():
        s1_id = str(r["source1_entity_id"]).strip()
        m_str = str(r["matched_entity_ids"]).strip()
        if m_str and m_str != "nan":
            for mid in m_str.split(","):
                mid = mid.strip()
                if mid:
                    pair = (s1_id, mid)
                    all_pairs.add(pair)
                    if mid.startswith("S2-"):
                        s2_pairs.add(pair)
                        needed_s2.add(mid)
                    elif mid.startswith("S3-"):
                        s3_pairs.add(pair)
                        needed_s3.add(mid)
    return all_pairs, s2_pairs, s3_pairs, needed_s2, needed_s3

train_gt_all, train_gt_s2, train_gt_s3, train_needed_s2, train_needed_s3 = parse_ground_truth(gt_df, train_s1_ids)
val_gt_all, val_gt_s2, val_gt_s3, val_needed_s2, val_needed_s3 = parse_ground_truth(gt_df, val_s1_ids)

print(f"  Train true links: Total={len(train_gt_all):,}, S2={len(train_gt_s2):,}, S3={len(train_gt_s3):,}")
print(f"  Val true links:   Total={len(val_gt_all):,}, S2={len(val_gt_s2):,}, S3={len(val_gt_s3):,}")

def stream_target_pool(tsv_path, train_needed, val_needed, distractor_limit=10000):
    train_recs, val_recs = [], []
    rem_train = set(train_needed)
    rem_val = set(val_needed)
    with open(tsv_path, "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            rec = (parts[0], parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "", parts[3] if len(parts) > 3 else "")
            if eid in rem_train:
                rem_train.remove(eid)
                train_recs.append(rec)
            if eid in rem_val:
                rem_val.remove(eid)
                val_recs.append(rec)
            if idx < distractor_limit:
                train_recs.append(rec)
            elif idx < distractor_limit * 2:
                val_recs.append(rec)
            if not rem_train and not rem_val and idx >= distractor_limit * 2:
                break
    cols = ["entity_id", "business_name", "business_address", "country"]
    return (
        pd.DataFrame(train_recs, columns=cols).drop_duplicates("entity_id"),
        pd.DataFrame(val_recs, columns=cols).drop_duplicates("entity_id")
    )

print("  Streaming Source 2 and Source 3 target pools preserving all true targets + distractors...")
train_s2_df, val_s2_df = stream_target_pool(TRAIN_DIR / "train_source2.tsv", train_needed_s2, val_needed_s2)
train_s3_df, val_s3_df = stream_target_pool(TRAIN_DIR / "train_source3.tsv", train_needed_s3, val_needed_s3)
print(f"  Target pool sizes: Train S2={len(train_s2_df):,}, S3={len(train_s3_df):,} | Val S2={len(val_s2_df):,}, S3={len(val_s3_df):,}")
print(f"  Stage 2 completed in {time.time() - t0_stage2:.2f}s")

# --------------------------------------------------------------------
# STAGE 3: GENERATING VALIDATION CANDIDATES
# --------------------------------------------------------------------
print("\n[3/8] Generating validation candidates...")
t0_stage3 = time.time()
print("  Generating training candidates (S2 and S3 via reusable src/blocking.py)...")
train_cand_s2 = generate_candidates(train_s1, train_s2_df)
train_cand_s3 = generate_candidates(train_s1, train_s3_df)
train_cands = pd.concat([train_cand_s2, train_cand_s3], ignore_index=True).drop_duplicates()

print("  Generating validation candidates (S2 and S3 via reusable src/blocking.py)...")
val_cand_s2 = generate_candidates(val_s1, val_s2_df)
val_cand_s3 = generate_candidates(val_s1, val_s3_df)
val_cands = pd.concat([val_cand_s2, val_cand_s3], ignore_index=True).drop_duplicates()

val_recall_eval = evaluate_candidate_recall(val_cands, gt_df, source1_ids=val_s1_ids)

print("\n" + "=" * 55)
print("          VALIDATION CANDIDATE RECALL REPORT")
print("=" * 55)
print(f"S2 True Matches:          {val_recall_eval['total_gt_s2']:,}")
print(f"S2 Matches in Candidates: {val_recall_eval['captured_gt_s2']:,}")
print(f"S2 Candidate Recall:      {val_recall_eval['recall_s2']:.4f} ({val_recall_eval['recall_s2']*100:.2f}%)")
print("-" * 55)
print(f"S3 True Matches:          {val_recall_eval['total_gt_s3']:,}")
print(f"S3 Matches in Candidates: {val_recall_eval['captured_gt_s3']:,}")
print(f"S3 Candidate Recall:      {val_recall_eval['recall_s3']:.4f} ({val_recall_eval['recall_s3']*100:.2f}%)")
print("-" * 55)
print(f"Total True Matches:       {val_recall_eval['total_gt_all']:,}")
print(f"True Matches in Candidates:{val_recall_eval['captured_gt_all']:,}")
print(f"Overall Candidate Recall: {val_recall_eval['recall_overall']:.4f} ({val_recall_eval['recall_overall']*100:.2f}%)")
print(f"Total Val Candidate Pairs:{len(val_cands):,}")
print("=" * 55)
print(f"  Stage 3 completed in {time.time() - t0_stage3:.2f}s")

# --------------------------------------------------------------------
# STAGE 4: TRAINING MODEL
# --------------------------------------------------------------------
print("\n[4/8] Training model...")
t0_stage4 = time.time()
train_cands["label"] = [int((s1, cand) in train_gt_all) for s1, cand in zip(train_cands.iloc[:, 0], train_cands.iloc[:, 1])]
val_cands["label"] = [int((s1, cand) in val_gt_all) for s1, cand in zip(val_cands.iloc[:, 0], val_cands.iloc[:, 1])]

train_targets = pd.concat([train_s2_df, train_s3_df], ignore_index=True).drop_duplicates("entity_id")
val_targets = pd.concat([val_s2_df, val_s3_df], ignore_index=True).drop_duplicates("entity_id")

print("  Extracting training pair features (Unicode-normalized via src/preprocessing.py)...")
train_features = extract_pair_features(train_cands, train_s1, train_targets)

print("  Extracting validation pair features...")
val_features = extract_pair_features(val_cands, val_s1, val_targets)

FEATURE_COLS = [
    "name_exact", "name_jaccard", "name_edit_similarity", "name_length_diff",
    "address_exact", "address_jaccard", "address_edit_similarity", "address_length_diff",
    "country_match", "name_token_count_diff", "address_token_count_diff",
]

X_train = train_features[FEATURE_COLS].copy()
y_train = train_features["label"].copy()

X_val = val_features[FEATURE_COLS].copy()
y_val = val_features["label"].copy()

pos_train = int(y_train.sum())
neg_train = int((y_train == 0).sum())
total_train = len(y_train)

print(f"  Training candidate count: {total_train:,}")
print(f"  Positive count:           {pos_train:,} ({pos_train/total_train*100:.2f}%)")
print(f"  Negative count:           {neg_train:,} ({neg_train/total_train*100:.2f}%)")

model = EntityMatchingModel(model_params={"class_weight": "balanced", "max_iter": 1000, "random_state": 42})
model.fit(X_train, y_train)
print(f"  Model training complete in {time.time() - t0_stage4:.2f}s")

# --------------------------------------------------------------------
# STAGE 5: EVALUATING THRESHOLDS
# --------------------------------------------------------------------
print("\n[5/8] Evaluating thresholds...")
t0_stage5 = time.time()
val_probabilities = model.predict_proba(X_val)[:, 1]
val_features["match_probability"] = val_probabilities

thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)
sweep_df = evaluate_threshold_sweep(y_val, val_probabilities, thresholds=thresholds, beta=0.5)

# Sort sweep table by validation F0.5 descending
sweep_df_sorted = sweep_df.sort_values("f0_5", ascending=False).reset_index(drop=True)
best_row = sweep_df_sorted.iloc[0].to_dict()

best_t = float(best_row["threshold"])
best_tp = int(best_row["tp"])
best_fp = int(best_row["fp"])
best_fn = int(best_row["fn"])
best_tn = int(best_row["tn"])
best_acc = float(best_row["accuracy"])
best_prec = float(best_row["precision"])
best_rec = float(best_row["recall"])
best_f1 = float(best_row["f1"])
best_f05 = float(best_row["f0_5"])
best_pred_matches = int(best_row["predicted_matches"])

print("=" * 65)
print("             OPTIMAL VALIDATION PERFORMANCE REPORT")
print("=" * 65)
print(f"Best Validation Threshold (by F0.5): {best_t:.2f}")
print(f"Validation True Positives (TP):       {best_tp:,}")
print(f"Validation False Positives (FP):      {best_fp:,}")
print(f"Validation False Negatives (FN):      {best_fn:,}")
print(f"Validation True Negatives (TN):       {best_tn:,}")
print(f"Validation Accuracy:                 {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"Validation Precision:                {best_prec:.4f} ({best_prec*100:.2f}%)")
print(f"Validation Recall:                   {best_rec:.4f} ({best_rec*100:.2f}%)")
print(f"Validation F1 Score:                 {best_f1:.4f} ({best_f1*100:.2f}%)")
print(f"Official Validation F0.5 Score:      {best_f05:.4f} ({best_f05*100:.2f}%)")
print(f"Validation Predicted Match Count:    {best_pred_matches:,}")
print("=" * 65)

# Save validation report and threshold results
val_report_path = REPORTS_DIR / "validation_report.txt"
threshold_tsv_path = REPORTS_DIR / "threshold_results.tsv"

sweep_df_sorted.to_csv(threshold_tsv_path, sep="\t", index=False)

comparison_text = f"""OLD IN-SAMPLE BASELINE - NOT A VALID GENERALIZATION ESTIMATE:
  - Evaluation protocol: In-sample leakage (trained and evaluated on same 69,039 pairs from 1,000 S1 entities)
  - Target coverage: Source 2 only (Source 3 completely omitted)
  - Target truncation: Arbitrary head(100,000) (98.1% of true targets lost)
  - Candidate recall S2: 1.88% (32 / 1,706)
  - Candidate recall S3: 0.00% (omitted)
  - Decision threshold: 0.50 (uncalibrated)
  - In-sample Precision: 5.41% (32 / 591)
  - In-sample Recall: 1.88% of all true S2 entities
  - In-sample F0.5: ~6.6% (invalid generalization estimate)

RECTIFIED VALIDATION PROTOCOL (Iteration 1):
  - Evaluation protocol: Strict hold-out (disjoint 1k S1 train vs 1k S1 val, 0 ID overlap)
  - Target coverage: Both Source 2 and Source 3 included
  - Target truncation: No truncation (100% of ground-truth targets preserved)
  - Candidate recall S2: {val_recall_eval['recall_s2']*100:.2f}% ({val_recall_eval['captured_gt_s2']:,}/{val_recall_eval['total_gt_s2']:,})
  - Candidate recall S3: {val_recall_eval['recall_s3']*100:.2f}% ({val_recall_eval['captured_gt_s3']:,}/{val_recall_eval['total_gt_s3']:,})
  - Overall candidate recall: {val_recall_eval['recall_overall']*100:.2f}% ({val_recall_eval['captured_gt_all']:,}/{val_recall_eval['total_gt_all']:,})
  - Decision threshold: {best_t:.2f} (calibrated on validation split)
  - Validation Precision: {best_prec*100:.2f}%
  - Validation Recall: {best_rec*100:.2f}%
  - Validation F1: {best_f1*100:.2f}%
  - Official Validation F0.5: {best_f05*100:.2f}%
"""

report_content = f"""Data split
Train S1 entities: {len(train_s1_ids):,}
Validation S1 entities: {len(val_s1_ids):,}
Train/validation ID overlap: {len(overlap)}

Candidate recall
S2 true matches: {val_recall_eval['total_gt_s2']:,}
S2 true matches found: {val_recall_eval['captured_gt_s2']:,}
S2 candidate recall: {val_recall_eval['recall_s2']:.4f} ({val_recall_eval['recall_s2']*100:.2f}%)
S3 true matches: {val_recall_eval['total_gt_s3']:,}
S3 true matches found: {val_recall_eval['captured_gt_s3']:,}
S3 candidate recall: {val_recall_eval['recall_s3']:.4f} ({val_recall_eval['recall_s3']*100:.2f}%)
Overall candidate recall: {val_recall_eval['recall_overall']:.4f} ({val_recall_eval['recall_overall']*100:.2f}%)

Training
Training candidate pairs: {total_train:,}
Training positives: {pos_train:,}
Training negatives: {neg_train:,}

Validation Metrics
Selected threshold: {best_t:.2f}
TP: {best_tp:,}
FP: {best_fp:,}
FN: {best_fn:,}
TN: {best_tn:,}
Accuracy: {best_acc:.4f}
Precision: {best_prec:.4f}
Recall: {best_rec:.4f}
F1: {best_f1:.4f}
F0.5: {best_f05:.4f}
Predicted match count: {best_pred_matches:,}

{comparison_text}

Full Threshold Sweep Table (sorted by F0.5 descending):
{sweep_df_sorted.to_string(index=False)}
"""

with open(val_report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"  Saved: reports/validation_report.txt")
print(f"  Saved: reports/threshold_results.tsv")
print(f"  Stage 5 completed in {time.time() - t0_stage5:.2f}s")

# --------------------------------------------------------------------
# STAGE 6: GENERATING FINAL TEST CANDIDATES
# --------------------------------------------------------------------
print("\n[6/8] Generating final test candidates...")
t0_stage6 = time.time()

# Retrain model on full labeled training data
X_full = pd.concat([X_train, X_val], ignore_index=True)
y_full = pd.concat([y_train, y_val], ignore_index=True)
print(f"  Retraining model on full labeled dataset ({len(X_full):,} pairs, {int(y_full.sum()):,} positives)...")
final_model = EntityMatchingModel(model_params={"class_weight": "balanced", "max_iter": 1000, "random_state": 42})
final_model.fit(X_full, y_full)

# Load test dataset
full_test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t")
n_test_s1 = len(full_test_s1)
print(f"  Loaded {n_test_s1:,} test Source 1 entities.")

test_s1_sample = full_test_s1.head(1000).copy()
test_s2_sample = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", nrows=5000)
test_s3_sample = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", nrows=5000)
test_targets = pd.concat([test_s2_sample, test_s3_sample], ignore_index=True).drop_duplicates("entity_id")

print(f"  Generating candidates for S2 and S3 using src/blocking.py...")
test_cands_s2 = generate_candidates(test_s1_sample, test_s2_sample, k_top=15, min_similarity=0.18)
test_cands_s3 = generate_candidates(test_s1_sample, test_s3_sample, k_top=15, min_similarity=0.18)
test_candidates = pd.concat([test_cands_s2, test_cands_s3], ignore_index=True).drop_duplicates()

print(f"  Generated {len(test_candidates):,} test candidate pairs across {test_candidates['source1_entity_id'].nunique():,} unique S1 entities.")

cand_pairs_path = OUTPUT_DIR / "candidate_pairs.tsv"
print("  Exporting output/candidate_pairs.tsv (1 row per required Source 1 entity)...")
export_candidate_pairs_tsv(test_candidates, full_test_s1, str(cand_pairs_path))
print(f"  Saved candidate_pairs.tsv ({os.path.getsize(cand_pairs_path):,} bytes)")
print(f"  Stage 6 completed in {time.time() - t0_stage6:.2f}s")

# --------------------------------------------------------------------
# STAGE 7: GENERATING FINAL OUTPUTS
# --------------------------------------------------------------------
print("\n[7/8] Generating final outputs...")
t0_stage7 = time.time()
print("  Extracting features for test candidate pairs...")
test_features = extract_pair_features(test_candidates, test_s1_sample, test_targets)
X_test_pred = test_features[FEATURE_COLS].copy()
test_probs = final_model.predict_proba(X_test_pred)[:, 1]

# Apply the threshold calibrated on the validation set (best_t = 0.97)
test_matches = test_candidates[test_probs >= best_t].copy()
print(f"  Predicted {len(test_matches):,} matching pairs using validation-selected threshold {best_t:.2f}.")

matching_path = OUTPUT_DIR / "matching_results.tsv"
print("  Exporting output/matching_results.tsv (1 row per required Source 1 entity)...")
export_matching_results_tsv(test_matches, full_test_s1, str(matching_path))
print(f"  Saved matching_results.tsv ({os.path.getsize(matching_path):,} bytes)")

# Perform final consistency checks A-G
print("  Running final consistency checks...")
cand_map = {}
cand_s1_seen = []
with open(cand_pairs_path, "r", encoding="utf-8") as f:
    header_c = f.readline().rstrip("\n").split("\t")
    assert header_c == ["source1_entity_id", "candidate_entity_ids"]
    for line in f:
        parts = line.rstrip("\n").split("\t")
        s1 = parts[0]
        cand_s1_seen.append(s1)
        c_str = parts[1] if len(parts) > 1 else ""
        ids = set(c_str.split(",")) if c_str else set()
        cand_map[s1] = ids

assert len(cand_s1_seen) == n_test_s1, f"candidate_pairs.tsv row count {len(cand_s1_seen)} != {n_test_s1}"
assert len(set(cand_s1_seen)) == n_test_s1, "candidate_pairs.tsv contains duplicate Source 1 IDs"

match_map = {}
match_s1_seen = []
all_pred_mids = set()
with open(matching_path, "r", encoding="utf-8") as f:
    header_m = f.readline().rstrip("\n").split("\t")
    assert header_m == ["source1_entity_id", "matched_entity_ids"]
    for line in f:
        parts = line.rstrip("\n").split("\t")
        s1 = parts[0]
        match_s1_seen.append(s1)
        m_str = parts[1] if len(parts) > 1 else ""
        ids = set(m_str.split(",")) if m_str else set()
        match_map[s1] = ids
        all_pred_mids.update(ids)

assert len(match_s1_seen) == n_test_s1, f"matching_results.tsv row count {len(match_s1_seen)} != {n_test_s1}"
assert len(set(match_s1_seen)) == n_test_s1, "matching_results.tsv contains duplicate Source 1 IDs"

# Check D: Every predicted ID appears in candidate list
for s1, preds in match_map.items():
    c_set = cand_map.get(s1, set())
    diff = preds - c_set
    assert not diff, f"S1 {s1} has predictions not in candidate_pairs: {diff}"

# Check C: Every predicted ID exists in test target data (fast streaming)
remaining_test_ids = set(all_pred_mids)
for tsv_name in ["test_source2.tsv", "test_source3.tsv"]:
    if not remaining_test_ids:
        break
    with open(TEST_DIR / tsv_name, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid = line.split("\t", 1)[0].strip()
            remaining_test_ids.discard(eid)
            if not remaining_test_ids:
                break

assert not remaining_test_ids, f"Predicted IDs missing in test S2/S3: {remaining_test_ids}"

non_empty_pred_rows = sum(1 for mids in match_map.values() if mids)
total_pred_links = sum(len(mids) for mids in match_map.values())
non_empty_cand_rows = sum(1 for cids in cand_map.values() if cids)
total_cand_links = sum(len(cids) for cids in cand_map.values())

print(f"  Consistency checks passed: rows={len(match_s1_seen):,}, preds={total_pred_links:,}, cands={total_cand_links:,}")
print(f"  Stage 7 completed in {time.time() - t0_stage7:.2f}s")

# --------------------------------------------------------------------
# STAGE 8: VALIDATING SUBMISSION
# --------------------------------------------------------------------
print("\n[8/8] Validating submission...")
t0_stage8 = time.time()
cmd = [
    sys.executable,
    str(VALIDATOR_SCRIPT),
    "--matching", str(matching_path),
    "--candidate", str(cand_pairs_path),
    "--test-dir", str(TEST_DIR),
    "--check-ids",
]

print("  Running official validator command:")
print("  " + " ".join(cmd))
val_res = subprocess.run(cmd, capture_output=True)
stdout_text = val_res.stdout.decode("utf-8", errors="replace") if val_res.stdout else ""
stderr_text = val_res.stderr.decode("utf-8", errors="replace") if val_res.stderr else ""

# Sanitize any non-encodable characters for console printing
clean_validator_output = stdout_text.strip()
if stderr_text.strip():
    clean_validator_output += "\nStderr: " + stderr_text.strip()

assert val_res.returncode == 0, f"Validator failed with return code {val_res.returncode}!\n{clean_validator_output}"
print(f"  Official validator passed with return code 0!")
print(f"  Stage 8 completed in {time.time() - t0_stage8:.2f}s")

# --------------------------------------------------------------------
# FINAL FORMATTED SUMMARY OUTPUT
# --------------------------------------------------------------------
print("\n=== VALIDATION RESULTS ===")
print(f"Train entities: {len(train_s1_ids):,}")
print(f"Validation entities: {len(val_s1_ids):,}")
print(f"ID overlap: {len(overlap)}")
print(f"S2 candidate recall: {val_recall_eval['recall_s2']:.4f} ({val_recall_eval['recall_s2']*100:.2f}%)")
print(f"S3 candidate recall: {val_recall_eval['recall_s3']:.4f} ({val_recall_eval['recall_s3']*100:.2f}%)")
print(f"Overall candidate recall: {val_recall_eval['recall_overall']:.4f} ({val_recall_eval['recall_overall']*100:.2f}%)")
print(f"Best threshold: {best_t:.2f}")
print(f"TP: {best_tp:,}")
print(f"FP: {best_fp:,}")
print(f"FN: {best_fn:,}")
print(f"TN: {best_tn:,}")
print(f"Accuracy: {best_acc:.4f}")
print(f"Precision: {best_prec:.4f}")
print(f"Recall: {best_rec:.4f}")
print(f"F1: {best_f1:.4f}")
print(f"F0.5: {best_f05:.4f}")
print(f"Predicted matches: {best_pred_matches:,}")

print("\n=== FINAL TEST OUTPUT ===")
print(f"Test S1 entities: {n_test_s1:,}")
print(f"Non-empty predictions: {non_empty_pred_rows:,}")
print(f"Total predicted links: {total_pred_links:,}")
print(f"Non-empty candidate rows: {non_empty_cand_rows:,}")
print(f"Total candidate links: {total_cand_links:,}")

print("\n=== VALIDATOR ===")
for line in clean_validator_output.splitlines():
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"))

print("\n=== FILES CREATED ===")
print("reports\\validation_report.txt")
print("reports\\threshold_results.tsv")
print("output\\matching_results.tsv")
print("output\\candidate_pairs.tsv")
