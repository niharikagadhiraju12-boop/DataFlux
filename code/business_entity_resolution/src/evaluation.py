"""
Evaluation Module for Business Entity Resolution.

Responsibilities:
- Implementing evaluation metrics for the entity matching pipeline
- Calculating the official challenge evaluation metric: F0.5 Score
- Evaluating blocking quality (Pair Completeness, Reduction Ratio)
- Tracking Precision, Recall, and F-beta across validation splits
- Decision threshold calibration and sweeping for F0.5 optimization

IMPORTANT CHALLENGE METRIC NOTE:
-------------------------------
The official challenge metric is F0.5.
Evaluation must strictly follow the official challenge README / problem statement.

The F-beta score with beta = 0.5 weights Precision higher than Recall:
    F0.5 = (1 + 0.5^2) * (Precision * Recall) / (0.5^2 * Precision + Recall)
         = 1.25 * (Precision * Recall) / (0.25 * Precision + Recall)

This reflects the critical requirement in business entity resolution to avoid
false positive matches (incorrectly merging distinct business entities) while
maintaining strong coverage.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd


def compute_f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute the F-beta score given precision and recall.
    Default beta is 0.5 (official challenge metric).

    Parameters:
        precision: Precision value [0.0, 1.0].
        recall: Recall value [0.0, 1.0].
        beta: Weighting factor, default 0.5.

    Returns:
        F-beta score as a float.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    f_beta = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return float(f_beta)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    beta: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Compute classification metrics: TP, FP, FN, TN, Precision, Recall, and F0.5.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        beta: F-score beta parameter (default 0.5).

    Returns:
        Dictionary containing tp, fp, fn, tn, precision, recall, f_beta, and predicted_matches.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    predicted_matches = tp + fp
    total_samples = tp + fp + fn + tn
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    precision = float(tp / predicted_matches) if predicted_matches > 0 else 0.0
    actual_positives = tp + fn
    recall = float(tp / actual_positives) if actual_positives > 0 else 0.0
    f_beta = compute_f_beta_score(precision, recall, beta=beta)
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_beta": f_beta,
        "f0_5": f_beta,
        "predicted_matches": predicted_matches,
    }


def evaluate_threshold_sweep(
    y_true: Any,
    y_prob: Any,
    thresholds: Optional[Sequence[float]] = None,
    beta: float = 0.5,
) -> pd.DataFrame:
    """
    Evaluate binary classification metrics across a range of decision thresholds.

    Parameters:
        y_true: Ground truth binary labels.
        y_prob: Predicted match probabilities in [0.0, 1.0].
        thresholds: Sequence of thresholds to evaluate (defaults to 0.50 to 0.99 in step 0.01).
        beta: F-score beta parameter (default 0.5).

    Returns:
        pd.DataFrame summarizing threshold metrics.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)

    y_t = np.asarray(y_true).astype(int)
    y_pr = np.asarray(y_prob).astype(float)

    rows = []
    for thresh in thresholds:
        thresh = float(thresh)
        y_p = (y_pr >= thresh).astype(int)
        metrics = evaluate_predictions(y_t, y_p, beta=beta)
        metrics["threshold"] = thresh
        rows.append(metrics)

    df_sweep = pd.DataFrame(rows)
    cols = [
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f0_5",
        "predicted_matches",
    ]
    return df_sweep[[c for c in cols if c in df_sweep.columns]]


def find_best_threshold(
    sweep_df: pd.DataFrame,
    metric: str = "f0_5",
) -> Dict[str, Any]:
    """
    Find the threshold that maximizes the specified evaluation metric (default f0_5).

    Parameters:
        sweep_df: DataFrame returned by evaluate_threshold_sweep.
        metric: Column name to maximize.

    Returns:
        Dict corresponding to the best row.
    """
    if sweep_df.empty:
        raise ValueError("Sweep DataFrame is empty.")

    best_idx = sweep_df[metric].idxmax()
    return sweep_df.loc[best_idx].to_dict()


def evaluate_candidate_recall(
    candidate_pairs: Union[pd.DataFrame, Set[Tuple[str, str]]],
    ground_truth: pd.DataFrame,
    source1_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate candidate recall broken down by target source (S2, S3, and overall).

    Parameters:
        candidate_pairs: DataFrame with ['source1_entity_id', 'candidate_entity_id'] or set of tuples.
        ground_truth: DataFrame with ['source1_entity_id', 'matched_entity_ids'].
        source1_ids: Optional filter on Source 1 IDs to evaluate.

    Returns:
        Dictionary of recall metrics for S2, S3, and overall.
    """
    if isinstance(candidate_pairs, pd.DataFrame):
        cand_set: Set[Tuple[str, str]] = set(
            zip(candidate_pairs.iloc[:, 0], candidate_pairs.iloc[:, 1])
        )
    else:
        cand_set = set(candidate_pairs)

    gt_df = ground_truth.copy()
    if source1_ids is not None:
        gt_df = gt_df[gt_df["source1_entity_id"].isin(source1_ids)]

    gt_all: Set[Tuple[str, str]] = set()
    gt_s2: Set[Tuple[str, str]] = set()
    gt_s3: Set[Tuple[str, str]] = set()

    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = str(row["matched_entity_ids"]).strip()
        if m_str and m_str != "nan":
            for mid in m_str.split(","):
                mid = mid.strip()
                if mid:
                    pair = (s1, mid)
                    gt_all.add(pair)
                    if mid.startswith("S2-"):
                        gt_s2.add(pair)
                    elif mid.startswith("S3-"):
                        gt_s3.add(pair)

    cap_all = len(gt_all.intersection(cand_set))
    cap_s2 = len(gt_s2.intersection(cand_set))
    cap_s3 = len(gt_s3.intersection(cand_set))

    recall_all = cap_all / max(1, len(gt_all))
    recall_s2 = cap_s2 / max(1, len(gt_s2))
    recall_s3 = cap_s3 / max(1, len(gt_s3))

    return {
        "total_gt_all": len(gt_all),
        "captured_gt_all": cap_all,
        "recall_overall": recall_all,
        "total_gt_s2": len(gt_s2),
        "captured_gt_s2": cap_s2,
        "recall_s2": recall_s2,
        "total_gt_s3": len(gt_s3),
        "captured_gt_s3": cap_s3,
        "recall_s3": recall_s3,
        "total_candidates": len(cand_set),
    }


def evaluate_blocking_performance(
    true_matches: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    total_comparisons: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute candidate generation metrics:
    - Pair Completeness (PC / Recall of true pairs retained by blocking)
    - Reduction Ratio (RR / fraction of Cartesian product pruned)
    """
    metrics = evaluate_candidate_recall(candidate_pairs, true_matches)
    rr = None
    if total_comparisons and total_comparisons > 0:
        rr = 1.0 - (metrics["total_candidates"] / total_comparisons)

    return {
        "pair_completeness": metrics["recall_overall"],
        "reduction_ratio": rr if rr is not None else 0.0,
        "total_candidates": metrics["total_candidates"],
        "captured_pairs": metrics["captured_gt_all"],
        "total_true_pairs": metrics["total_gt_all"],
    }
