# Blocking / Candidate Generation — P1

Generates `candidate_pairs.tsv`: for every Source-1 entity, the plausible S2/S3
matches from the blocking stage, before the matching model narrows them down.

## Files
- `normalize.py` — name/address normalization (legal-suffix stripping, order-
  invariant canonical form, transliteration approximation via `unidecode`,
  address abbreviation expansion, pincode/state extraction)
- `build_candidates.py` — the blocking pipeline (see below)
- `evaluate_recall.py` — measures pair-level recall + reduction ratio against
  `train_ground_truth.tsv`

## Approach
Five independent blocking signals, unioned (a pair only needs to hit ONE):

| Signal | What it catches |
|---|---|
| `name_prefix` | word-order transpositions, minor tail typos |
| `name_token` | one distinctive shared word even if the rest of the name is noisy |
| `name_concat` | domain-style names (`libertyfamilyoffice.com` vs "Liberty Family Office") |
| `addr_pincode` | exact address match via postal code |
| `addr_token` | **the fallback for non-Latin-script names** — addresses in this dataset stay in Latin script even when the business name is transliterated (Hindi/Kannada/Telugu/etc.), so address tokens still line up when name tokens can't |

Each record is indexed under its **rarest 2–3 tokens** (by in-source document
frequency), not every token — this is what keeps block sizes bounded at
millions of rows instead of exploding on common words. Tokens more common than
`MAX_TOKEN_BLOCK` (5000 records) are dropped outright as non-discriminative.

Candidates are deduplicated, scored with a cheap similarity function
(`0.7 × rapidfuzz.token_sort_ratio(name) + 0.3 × address_token_jaccard × 100`),
and capped to the top `TOP_K=30` per Source-1 entity.

**Why not use unidecode-transliterated names for name matching directly?**
Tried it — `unidecode` is phonetic-per-character, not word-aware, so
`रेड वेंचर्स प्राइवेट लिमिटेड` → `redd veNcrs praaivett limittedd`, which
doesn't token-match `red ventures private limited` even loosely. It's kept
only as a fallback signal generator (first-token overlap occasionally still
works); the `addr_token` signal is what actually recovers these cases.

## Current results (sample validation)
Measured on a 2,000-entity S1 slice against a 300K-row S2/S3 sample
(so these numbers are from a reduced search universe — see note below):

- **Pair-level recall: 90.3%**
- **Entity fully-recalled rate: 89.4%**
- Avg candidates per S1 entity: ~30 (capped)
- Reduction ratio: 0.99995

Run it yourself:
```bash
python3 build_candidates.py \
  --source1 /path/train_source1.tsv \
  --source2 /path/train_source2.tsv \
  --source3 /path/train_source3.tsv \
  --out candidate_pairs.tsv \
  --limit 2000          # drop this flag for a full run

python3 evaluate_recall.py \
  --candidates candidate_pairs.tsv \
  --ground-truth /path/train_ground_truth.tsv \
  --source2 /path/train_source2.tsv \
  --source3 /path/train_source3.tsv
```

## Scaling to the full dataset (2.2M S1 / 5.0M S2 / 5.3M S3)
This was validated on a reduced sample; a full run needs to happen offline
(it exceeds a few minutes, so run it as a background job, not interactively).

**Use `--country` to partition the run** — matches never cross countries, so
this is a lossless way to bound memory/runtime:
```bash
python3 build_candidates.py --source1 ... --source2 ... --source3 ... \
  --out candidate_pairs_us.tsv --country US

python3 build_candidates.py --source1 ... --source2 ... --source3 ... \
  --out candidate_pairs_india.tsv --country India

# then concatenate (keep header once):
head -1 candidate_pairs_us.tsv > candidate_pairs.tsv
tail -n +2 candidate_pairs_us.tsv >> candidate_pairs.tsv
tail -n +2 candidate_pairs_india.tsv >> candidate_pairs.tsv
```
The **test set adds France** — run a third pass with `--country France` on
the test files (no separate code change needed; country is read from the
data, nothing is hardcoded to `{US, India}`).

**Extrapolated runtime** (measured: normalization ~10.6s/300K rows,
scoring ~288K pairs/sec after vectorization):
- Normalization: ~3–4 min per source at full India-partition scale, more for
  the larger US partition — budget ~15–20 min total across both countries
- Candidate join + scoring: a few minutes per country partition
- **Rough total: 30–45 min for a full run**, run in the background/`nohup`

**If you hit memory pressure** on the full US partition (the larger one),
tune these constants at the top of `build_candidates.py`:
- Lower `MAX_TOKEN_BLOCK` (5000 → 2000) to drop more common tokens
- Lower `RAREST_K_NAME` / `RAREST_K_ADDR` (3/2 → 2/1) to index fewer keys per record
- Both trade a small amount of recall for a large reduction in raw pair count

## Known remaining misses (for P2/P3 awareness)
From manual inspection of the ~10% still missed:
- Single-character typos in **short** names (e.g. "Helios" vs "Helos") where
  no other signal saves the pair — a char-n-gram/prefix signal would help,
  not yet added
- Cases where the name is essentially unrelated but the address is identical
  (data quality issue in source records, not a blocking gap) — `addr_token`
  catches most but not all of these depending on how much address text survives
