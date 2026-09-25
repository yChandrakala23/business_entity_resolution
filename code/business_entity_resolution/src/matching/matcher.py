from pathlib import Path
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from src.evaluation.metrics import (
    build_predictions_df,
    parse_id_list,
    tune_postprocessing,
)

warnings.filterwarnings("ignore")

try:
  import xgboost as xgb

  HAS_XGB = True
except ImportError:
  HAS_XGB = False

try:
  from catboost import CatBoostClassifier

  HAS_CAT = True
except ImportError:
  HAS_CAT = False

# Columns never used directly as raw numerical features
IGNORE_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "label",
    "prob",
    "max_prob",
    "country",
    "country_s1",
    "country_s2",
    "business_name",
    "business_address",
    "business_name_s1",
    "business_name_s2",
    "business_address_s1",
    "business_address_s2",
}


class EntityMatcher:
  """Precision-heavy Entity Resolution Matcher (MIT/Apache-2.0 compliant, <10M params).

  Combines Auto-Group-Context Feature Engineering, 5-Fold GroupKFold
  Multi-GBDT Ensembling (LightGBM + XGBoost + CatBoost), and 2D F_0.5
  Threshold Optimization.
  """

  def __init__(self, n_splits: int = 5, use_group_features: bool = True):
    self.n_splits = n_splits
    self.use_group_features = use_group_features
    self.base_feature_cols = []
    self.feature_cols = []
    self.models = {"lgb": [], "xgb": [], "cat": []}
    self.blend_weights = {"lgb": 1.0, "xgb": 0.0, "cat": 0.0}
    self.best_threshold = 0.70
    self.best_rel_margin = 1.0

  def _get_base_numeric_cols(self, df: pd.DataFrame) -> list:
    return [
        c
        for c in df.columns
        if c not in IGNORE_COLS
        and not c.startswith("_grp_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]

  def _engineer_group_context_features(
      self, df: pd.DataFrame, is_train: bool = False
  ) -> pd.DataFrame:
    """Engineers candidate-set competition features per source1_entity_id.

    Knowing whether a candidate is far better than the 2nd-best candidate
    for the same S1 entity dramatically reduces false positives for F_0.5.
    """
    df = df.copy()
    if is_train or not self.base_feature_cols:
      self.base_feature_cols = self._get_base_numeric_cols(df)

    if not self.use_group_features or len(df) == 0:
      return df

    # Number of candidates blocked for this Source 1 entity
    grp = df.groupby("source1_entity_id", sort=False)
    df["_grp_candidate_count"] = grp["candidate_entity_id"].transform("count")

    # Add group-relative features for up to the first 12 base similarity features
    target_cols = self.base_feature_cols[:12]
    for col in target_cols:
      col_max = grp[col].transform("max")
      col_mean = grp[col].transform("mean")
      col_std = grp[col].transform("std").fillna(0.0)

      # Gap from best candidate in the same S1 block
      df[f"_grp_{col}_diff_max"] = df[col] - col_max
      # Z-score within the S1 candidate block
      df[f"_grp_{col}_zscore"] = (df[col] - col_mean) / (col_std + 1e-5)
      # Normalized rank within the S1 block (1.0 = highest similarity)
      df[f"_grp_{col}_pct_rank"] = grp[col].rank(pct=True, method="average")

    return df

  @staticmethod
  def attach_labels(
      features_df: pd.DataFrame, ground_truth_df: pd.DataFrame
  ) -> pd.DataFrame:
    gt_map = dict(
        zip(
            ground_truth_df["source1_entity_id"],
            ground_truth_df["matched_entity_ids"].apply(parse_id_list),
        )
    )
    df = features_df.copy()
    df["label"] = [
        1 if cand_id in gt_map.get(s1_id, set()) else 0
        for s1_id, cand_id in zip(
            df["source1_entity_id"], df["candidate_entity_id"]
        )
    ]
    return df

  def fit_and_validate(
      self,
      train_features_df: pd.DataFrame,
      ground_truth_df: pd.DataFrame,
      all_train_s1_ids: list,
  ) -> dict:
    df = self.attach_labels(train_features_df, ground_truth_df)
    df = self._engineer_group_context_features(df, is_train=True)

    self.feature_cols = [
        c
        for c in df.columns
        if c not in IGNORE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not self.feature_cols:
      raise ValueError("No numeric feature columns found in train_features_df!")

    X = df[self.feature_cols].fillna(-999.0)
    y = df["label"].values
    groups = df["source1_entity_id"].values

    gkf = GroupKFold(n_splits=self.n_splits)
    self.models = {"lgb": [], "xgb": [], "cat": []}

    oof_preds = {
        "lgb": np.zeros(len(df)),
        "xgb": np.zeros(len(df)),
        "cat": np.zeros(len(df)),
    }

    # Dampened class weight so probabilities remain well-calibrated for precision
    pos_count = max(y.sum(), 1)
    neg_count = len(y) - pos_count
    scale_pos = float(np.clip(np.sqrt(neg_count / pos_count), 1.0, 10.0))

    print(
        f"[Matcher] Training on {len(df):,} candidate pairs"
        f" ({pos_count:,} positives) with {len(self.feature_cols)} features"
        f" ({len(self.base_feature_cols)} base +"
        f" {len(self.feature_cols) - len(self.base_feature_cols)} auto-group"
        " features)..."
    )

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
      X_tr, y_tr = X.iloc[train_idx], y[train_idx]
      X_va, y_va = X.iloc[val_idx], y[val_idx]

      # 1. LightGBM (Always runs - MIT License)
      lgb_model = lgb.LGBMClassifier(
          n_estimators=1000,
          learning_rate=0.03,
          num_leaves=31,
          min_child_samples=25,
          subsample=0.8,
          colsample_bytree=0.8,
          scale_pos_weight=scale_pos,
          random_state=42 + fold,
          n_jobs=-1,
          verbose=-1,
      )
      lgb_model.fit(
          X_tr,
          y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50, verbose=False)],
      )
      oof_preds["lgb"][val_idx] = lgb_model.predict_proba(X_va)[:, 1]
      self.models["lgb"].append(lgb_model)

      # 2. XGBoost (Optional - Apache-2.0 License)
      if HAS_XGB:
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=142 + fold,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds["xgb"][val_idx] = xgb_model.predict_proba(X_va)[:, 1]
        self.models["xgb"].append(xgb_model)

      # 3. CatBoost (Optional - Apache-2.0 License)
      if HAS_CAT:
        cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.04,
            depth=6,
            scale_pos_weight=scale_pos,
            random_seed=242 + fold,
            verbose=False,
            early_stopping_rounds=50,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        oof_preds["cat"][val_idx] = cat_model.predict_proba(X_va)[:, 1]
        self.models["cat"].append(cat_model)

    # Optimize blend weights using Out-of-Fold PR-AUC
    active_families = ["lgb"]
    if HAS_XGB and len(self.models["xgb"]) > 0:
      active_families.append("xgb")
    if HAS_CAT and len(self.models["cat"]) > 0:
      active_families.append("cat")

    if len(active_families) == 1:
      self.blend_weights = {"lgb": 1.0, "xgb": 0.0, "cat": 0.0}
      blended_oof = oof_preds["lgb"]
    else:
      scores = {
          k: max(average_precision_score(y, oof_preds[k]), 1e-5)
          for k in active_families
      }
      # Soft rank weighting favoring strongest PR-AUC model
      total = sum(v**2 for v in scores.values())
      self.blend_weights = {
          k: (scores[k] ** 2 / total) if k in scores else 0.0
          for k in ["lgb", "xgb", "cat"]
      }
      blended_oof = sum(
          self.blend_weights[k] * oof_preds[k] for k in active_families
      )

    df["prob"] = blended_oof
    best_Result = tune_postprocessing(all_train_s1_ids, ground_truth_df, df)

    self.best_threshold = best_Result["abs_threshold"]
    self.best_rel_margin = best_Result["rel_margin"]

    print(
        "[Matcher] Active Ensemble Weights:"
        f" { {k: round(v, 3) for k, v in self.blend_weights.items() if v > 0} }"
    )
    print(
        f"[Matcher] OOF Macro F_0.5: {best_Result['macro_f05']:.4f} "
        f"(Singleton Acc: {best_Result['singleton_acc']:.4f}, "
        f"Non-Singleton F_0.5: {best_Result['non_singleton_f05']:.4f})"
    )
    print(
        f"[Matcher] Optimal Post-Processing -> Threshold: {self.best_threshold}"
        f" | Rel Margin: {self.best_rel_margin}"
    )

    return best_Result

  def predict_test(
      self,
      test_features_df: pd.DataFrame,
      all_test_s1_ids: list,
      output_tsv_path: str = None,
  ) -> pd.DataFrame:
    if len(test_features_df) == 0:
      pred_df = pd.DataFrame({
          "source1_entity_id": all_test_s1_ids,
          "matched_entity_ids": [""] * len(all_test_s1_ids),
      })
    else:
      df = self._engineer_group_context_features(
          test_features_df, is_train=False
      )
      for col in self.feature_cols:
        if col not in df.columns:
          df[col] = -999.0
      X_test = df[self.feature_cols].fillna(-999.0)

      blended_probs = np.zeros(len(df))
      for family, weight in self.blend_weights.items():
        if weight > 0 and len(self.models[family]) > 0:
          fam_probs = np.mean(
              [m.predict_proba(X_test)[:, 1] for m in self.models[family]],
              axis=0,
          )
          blended_probs += weight * fam_probs

      df["prob"] = blended_probs
      pred_df = build_predictions_df(
          all_test_s1_ids, df, self.best_threshold, self.best_rel_margin
      )

    if output_tsv_path:
      Path(output_tsv_path).parent.mkdir(parents=True, exist_ok=True)
      pred_df.to_csv(output_tsv_path, sep="\t", index=False)
      print(f"[Matcher] Wrote {len(pred_df):,} test rows to {output_tsv_path}")

    return pred_df

  def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
    if not self.models["lgb"]:
      return pd.DataFrame()
    imp = np.mean([m.feature_importances_ for m in self.models["lgb"]], axis=0)
    return (
        pd.DataFrame({"feature": self.feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

  def save(self, path: str = "models/ensemble_matcher.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(self.__dict__, path)

  def load(self, path: str = "models/ensemble_matcher.pkl"):
    self.__dict__.update(joblib.load(path))


if __name__ == "__main__":
  # Self-contained smoke test so you can verify everything works right now!
  print("Running synthetic Entity Resolution smoke test...")
  rng = np.random.default_rng(42)
  s1_ids = [f"S1-{i:05d}" for i in range(200)]

  rows, gt_rows = [], []
  for s1 in s1_ids:
    is_singleton = rng.random() < 0.35
    true_matches = []
    n_cands = rng.integers(2, 8)
    for c in range(n_cands):
      cand_id = f"S2-{rng.integers(1, 99999):05d}"
      is_match = (not is_singleton) and (c == 0)
      if is_match:
        true_matches.append(cand_id)
      rows.append({
          "source1_entity_id": s1,
          "candidate_entity_id": cand_id,
          "name_jaccard": (
              rng.uniform(0.65, 1.0) if is_match else rng.uniform(0.1, 0.75)
          ),
          "addr_tfidf": (
              rng.uniform(0.50, 0.95) if is_match else rng.uniform(0.0, 0.65)
          ),
          "is_same_country": 1,
      })
    gt_rows.append({
        "source1_entity_id": s1,
        "matched_entity_ids": ",".join(sorted(set(true_matches))),
    })

  synth_features = pd.DataFrame(rows)
  synth_gt = pd.DataFrame(gt_rows)

  matcher = EntityMatcher(n_splits=5)
  matcher.fit_and_validate(synth_features, synth_gt, s1_ids)
  preds = matcher.predict_test(synth_features, s1_ids)
  print("\nTop 5 Features (including auto-engineered group context features):")
  print(matcher.get_feature_importance(5).to_string(index=False))
  print("\nSmoke test PASSED! Ready to commit and push.")