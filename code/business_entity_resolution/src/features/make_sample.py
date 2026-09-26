"""
Builds a small, realistic candidates.tsv from the real train data so we can
smoke-test build_features.py before Person 1's actual blocking/normalize.py
exist. This is NOT the deliverable — it's a throwaway harness for P2's own
testing. Delete or keep out of the repo (or keep under scripts/dev/ if you
want it for future debugging).
"""
import random
import pandas as pd

random.seed(0)
N_S1 = 3000
NEG_PER_S1 = 3  # extra random (likely-wrong) candidates per S1 entity, to
                 # simulate blocking noise / non-matches

import os
os.chdir("/mnt/user-data/uploads")
OUT = "/home/claude"

s1 = pd.read_csv("train_source1.tsv", sep="\t", nrows=N_S1, dtype=str)
gt = pd.read_csv("train_ground_truth.tsv", sep="\t", dtype=str)
gt_sample = gt[gt["source1_entity_id"].isin(s1["entity_id"])].copy()
gt_sample["matched_entity_ids"] = gt_sample["matched_entity_ids"].fillna("")

# all true-positive candidate ids we need to pull out of source2/source3
true_ids = set()
for ids in gt_sample["matched_entity_ids"]:
    if ids:
        true_ids.update(ids.split(","))
s2_true = {i for i in true_ids if i.startswith("S2-")}
s3_true = {i for i in true_ids if i.startswith("S3-")}
print(f"S1 sample: {len(s1)}  true S2 matches: {len(s2_true)}  true S3 matches: {len(s3_true)}")

# stream-scan source2/source3 once each, grabbing true-positive rows + a
# random negative pool, without loading the full 5M-row files as a whole
def sample_source(path, true_ids, n_negative_pool):
    true_rows = []
    neg_pool = []
    reader = pd.read_csv(path, sep="\t", dtype=str, chunksize=200_000)
    for chunk in reader:
        hit = chunk[chunk["entity_id"].isin(true_ids)]
        if len(hit):
            true_rows.append(hit)
        if len(neg_pool) < n_negative_pool:
            neg_pool.append(chunk.sample(min(50, len(chunk)), random_state=1))
        if len(true_rows and pd.concat(true_rows)) >= len(true_ids) and len(neg_pool) * 50 >= n_negative_pool:
            break
    true_df = pd.concat(true_rows, ignore_index=True) if true_rows else chunk.iloc[0:0]
    neg_df = pd.concat(neg_pool, ignore_index=True).drop_duplicates(subset="entity_id") if neg_pool else chunk.iloc[0:0]
    return true_df, neg_df

s2_true_df, s2_neg_df = sample_source("train_source2.tsv", s2_true, N_S1 * NEG_PER_S1)
s3_true_df, s3_neg_df = sample_source("train_source3.tsv", s3_true, N_S1 * NEG_PER_S1)

source2_sample = pd.concat([s2_true_df, s2_neg_df]).drop_duplicates(subset="entity_id")
source3_sample = pd.concat([s3_true_df, s3_neg_df]).drop_duplicates(subset="entity_id")

# build candidates.tsv: true positives (label=1) + random negatives (label=0)
# signal/score are placeholders standing in for Person 1's real blocking output
rows = []
neg_ids_pool = list(source2_sample["entity_id"]) + list(source3_sample["entity_id"])
for _, r in s1.iterrows():
    s1_id = r["entity_id"]
    matched = gt_sample.loc[gt_sample["source1_entity_id"] == s1_id, "matched_entity_ids"]
    matched_ids = matched.iloc[0].split(",") if len(matched) and matched.iloc[0] else []
    for mid in matched_ids:
        rows.append({"source1_entity_id": s1_id, "candidate_entity_id": mid, "signal": "true_match", "score": 0.9, "label": 1})
    negs = random.sample(neg_ids_pool, min(NEG_PER_S1, len(neg_ids_pool)))
    for nid in negs:
        if nid not in matched_ids:
            rows.append({"source1_entity_id": s1_id, "candidate_entity_id": nid, "signal": "random_neg", "score": 0.2, "label": 0})

candidates = pd.DataFrame(rows)
s1.to_csv(f"{OUT}/sample_source1.tsv", sep="\t", index=False)
source2_sample.to_csv(f"{OUT}/sample_source2.tsv", sep="\t", index=False)
source3_sample.to_csv(f"{OUT}/sample_source3.tsv", sep="\t", index=False)
candidates.to_csv(f"{OUT}/sample_candidates.tsv", sep="\t", index=False)
print(f"candidates: {len(candidates)} rows ({candidates['label'].sum()} positive)")
print(f"source2 sample: {len(source2_sample)}  source3 sample: {len(source3_sample)}")
