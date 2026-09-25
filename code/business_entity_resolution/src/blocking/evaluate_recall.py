"""
Evaluate blocking quality: pairs-completeness (recall) and reduction ratio,
restricted to the S1 entities and S2/S3 universe actually present in the run
being evaluated (important when testing against a trimmed sample).
"""
import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)


def load_ground_truth(path, s1_ids, s2s3_universe):
    gt = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1id, matches = row
            if s1id not in s1_ids:
                continue
            ids = set(i for i in matches.split(",") if i) if matches.strip() else set()
            # restrict to matches that exist in the evaluated universe, so a
            # trimmed sample doesn't get unfairly penalized for IDs we excluded
            ids = {i for i in ids if i in s2s3_universe}
            gt[s1id] = ids
    return gt


def load_candidates(path):
    cand = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1id, ids = row[0], row[1] if len(row) > 1 else ""
            cand[s1id] = set(i for i in ids.split(",") if i) if ids.strip() else set()
    return cand


def load_ids(path, col=0):
    ids = set()
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            ids.add(row[col])
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    args = ap.parse_args()

    candidates = load_candidates(args.candidates)
    s1_ids = set(candidates.keys())
    s2_ids = load_ids(args.source2)
    s3_ids = load_ids(args.source3)
    universe = s2_ids | s3_ids

    gt = load_ground_truth(args.ground_truth, s1_ids, universe)

    total_true = 0
    total_found = 0
    entities_with_matches = 0
    entities_fully_recalled = 0
    total_candidates = 0

    for s1id in s1_ids:
        true_matches = gt.get(s1id, set())
        found = candidates.get(s1id, set())
        total_candidates += len(found)
        if true_matches:
            entities_with_matches += 1
            hit = true_matches & found
            total_true += len(true_matches)
            total_found += len(hit)
            if len(hit) == len(true_matches):
                entities_fully_recalled += 1

    pair_recall = total_found / total_true if total_true else float("nan")
    entity_recall = entities_fully_recalled / entities_with_matches if entities_with_matches else float("nan")
    avg_candidates = total_candidates / len(s1_ids) if s1_ids else 0
    reduction_ratio = 1 - (avg_candidates / len(universe)) if universe else float("nan")

    print(f"S1 entities evaluated:          {len(s1_ids):,}")
    print(f"S1 entities with true matches:  {entities_with_matches:,}")
    print(f"Total true match pairs:         {total_true:,}")
    print(f"Total true pairs recovered:     {total_found:,}")
    print(f"Pair-level recall:              {pair_recall:.4f}")
    print(f"Entity fully-recalled rate:     {entity_recall:.4f}  (all true matches present in candidates)")
    print(f"Avg candidates per S1 entity:   {avg_candidates:.1f}")
    print(f"Reduction ratio:                {reduction_ratio:.6f}")
