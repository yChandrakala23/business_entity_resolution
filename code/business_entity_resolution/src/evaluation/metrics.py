import numpy as np
import pandas as pd


def parse_id_list(val) -> set:
  if pd.isna(val) or not str(val).strip():
    return set()
  return set(x.strip() for x in str(val).split(",") if x.strip())


def evaluate_detailed_f05(
    all_s1_ids: list, ground_truth_df: pd.DataFrame, pred_df: pd.DataFrame
) -> dict:
  """Computes exact Macro F_0.5 across all Source 1 entities, plus singleton

  and non-singleton diagnostics to guide precision tuning.
  """
  gt_map = dict(
      zip(
          ground_truth_df["source1_entity_id"],
          ground_truth_df["matched_entity_ids"].apply(parse_id_list),
      )
  )
  pred_map = dict(
      zip(
          pred_df["source1_entity_id"],
          pred_df["matched_entity_ids"].apply(parse_id_list),
      )
  )

  all_scores = []
  singleton_scores = []
  non_singleton_scores = []

  for s1_id in all_s1_ids:
    true_set = gt_map.get(s1_id, set())
    pred_set = pred_map.get(s1_id, set())

    if len(true_set) == 0:
      s_score = 1.0 if len(pred_set) == 0 else 0.0
      all_scores.append(s_score)
      singleton_scores.append(s_score)
      continue

    if len(pred_set) == 0:
      all_scores.append(0.0)
      non_singleton_scores.append(0.0)
      continue

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (0.25 * prec + rec) == 0:
      f05 = 0.0
    else:
      f05 = (1.25 * prec * rec) / (0.25 * prec + rec)

    all_scores.append(f05)
    non_singleton_scores.append(f05)

  return {
      "macro_f05": float(np.mean(all_scores)) if all_scores else 0.0,
      "singleton_acc": (
          float(np.mean(singleton_scores)) if singleton_scores else 0.0
      ),
      "non_singleton_f05": (
          float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0
      ),
  }


def compute_macro_f05(
    all_s1_ids: list, ground_truth_df: pd.DataFrame, pred_df: pd.DataFrame
) -> float:
  return evaluate_detailed_f05(all_s1_ids, ground_truth_df, pred_df)[
      "macro_f05"
  ]


def build_predictions_df(
    all_s1_ids: list,
    candidates_df: pd.DataFrame,
    abs_threshold: float,
    rel_margin: float = 1.0,
) -> pd.DataFrame:
  """Filters candidate pairs using both an absolute probability threshold

  and a relative margin from the top candidate per Source 1 entity.
  Guarantees exact matching_results.tsv formatting.
  """
  if len(candidates_df) == 0:
    return pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": [""] * len(all_s1_ids),
    })

  df = candidates_df[
      ["source1_entity_id", "candidate_entity_id", "prob"]
  ].copy()
  max_prob_per_s1 = df.groupby("source1_entity_id")["prob"].transform("max")

  # Keep candidate only if it passes abs_threshold AND is within rel_margin of the best candidate for that S1
  mask = (df["prob"] >= abs_threshold) & (
      (max_prob_per_s1 - df["prob"]) <= rel_margin
  )
  pos_df = df[mask]

  grouped = (
      pos_df.groupby("source1_entity_id")["candidate_entity_id"]
      .apply(lambda ids: ",".join(sorted(set(ids))))
      .to_dict()
  )

  return pd.DataFrame({
      "source1_entity_id": all_s1_ids,
      "matched_entity_ids": [grouped.get(s1_id, "") for s1_id in all_s1_ids],
  })


def tune_postprocessing(
    all_s1_ids: list, ground_truth_df: pd.DataFrame, val_preds_df: pd.DataFrame
) -> dict:
  """Fast 2D grid search over (abs_threshold, rel_margin) to maximize Macro F_0.5."""
  gt_map = dict(
      zip(
          ground_truth_df["source1_entity_id"],
          ground_truth_df["matched_entity_ids"].apply(parse_id_list),
      )
  )

  # Pre-group candidates by S1 for 20x faster grid search
  df = val_preds_df[["source1_entity_id", "candidate_entity_id", "prob"]].copy()
  df["max_prob"] = df.groupby("source1_entity_id")["prob"].transform("max")

  best_params = {"abs_threshold": 0.65, "rel_margin": 1.0, "macro_f05": -1.0}

  # Stage 1: Coarse search over absolute threshold
  for thresh in np.arange(0.30, 0.94, 0.02):
    pred_df = build_predictions_df(all_s1_ids, df, float(thresh), 1.0)
    score = compute_macro_f05(all_s1_ids, ground_truth_df, pred_df)
    if score > best_params["macro_f05"]:
      best_params = {
          "abs_threshold": float(round(thresh, 3)),
          "rel_margin": 1.0,
          "macro_f05": score,
      }

  # Stage 2: Fine search around best threshold + relative margin from top-1 candidate
  center_t = best_params["abs_threshold"]
  fine_thresholds = np.clip(
      np.arange(center_t - 0.06, center_t + 0.065, 0.01), 0.15, 0.98
  )
  rel_margins = [1.0, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05]

  for thresh in fine_thresholds:
    for margin in rel_margins:
      pred_df = build_predictions_df(
          all_s1_ids, df, float(thresh), float(margin)
      )
      score = compute_macro_f05(all_s1_ids, ground_truth_df, pred_df)
      if score > best_params["macro_f05"]:
        best_params = {
            "abs_threshold": float(round(thresh, 3)),
            "rel_margin": float(margin),
            "macro_f05": score,
        }

  final_pred_df = build_predictions_df(
      all_s1_ids, df, best_params["abs_threshold"], best_params["rel_margin"]
  )
  details = evaluate_detailed_f05(all_s1_ids, ground_truth_df, final_pred_df)
  best_params.update(details)
  return best_params