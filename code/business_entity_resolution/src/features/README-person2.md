# Person 2 — Feature Engineering — status

## What's here
- `build_features.py` — the contracted function, drop into `src/features/build_features.py`
- `make_sample.py` — dev-only harness, builds a realistic small candidate set
  from the real train files (true matches from ground truth + random negatives)
  so you can smoke-test without the full 2.2M-row dataset or Person 1's
  real blocking output. Not part of the submission.
- `requirements-additions.txt` — append these to the shared `requirements.txt`

## Verified against real data
Ran `make_sample.py` on the real `train_source1/2/3.tsv` + `train_ground_truth.tsv`
(3,000 S1 entities, ~19k candidate pairs: true matches + random negatives).
Every feature separates true matches from random negatives cleanly, e.g.:

| feature | random negative (mean) | true match (mean) |
|---|---|---|
| name Jaccard | 0.002 | 0.680 |
| name TF-IDF cosine | 0.002 | 0.739 |
| address TF-IDF cosine | 0.006 | 0.756 |
| name Levenshtein | 0.172 | 0.777 |
| name phonetic match | 0.001 | 0.770 |

Confirmed the real singleton rate independently: 5.58% (matches the
teammate's ~5.6% claim) via `train_ground_truth.tsv`.

## Known gaps / open items for you or the team
1. **`src/blocking/normalize.py` doesn't exist yet in what I have** — I
   built a fallback normalizer (unidecode + legal-suffix stripping + a
   pincode regex) inside `build_features.py` so it runs standalone. The
   real file exists (or will) at `src/blocking/normalize.py` — once it's
   there, `build_features.py` auto-imports it and ignores the fallback.
   Sanity-check the fallback's assumed return shape (`_unpack_name` /
   `_unpack_addr`) against the real one when it lands.
2. **No pincode data in the real addresses I sampled** — US and India rows
   in the actual files don't carry a clean postal code (see samples in
   `train_source2.tsv`). The pincode-match feature will mostly be 0/absent;
   kept it in since it's free and occasionally fires, but don't rely on it.
3. **Levenshtein/Jaro-Winkler use rapidfuzz in a loop, not sparse-matrix
   vectorization** — genuinely can't be expressed as matrix algebra. Fine
   at 19k pairs (seconds); flag to Person 3/4 if it's slow at the real
   ~2.2M x 100 scale — the fix then is chunking + multiprocessing, not a
   different library.
4. Real `candidate_pairs.tsv` from Person 1's blocking hasn't been shared
   yet — `make_sample.py` fakes one from ground truth for testing. Swap in
   her real file the moment it exists; the column names it must contain
   (`source1_entity_id`, `candidate_entity_id`) are already what
   `build_features.py` expects.
