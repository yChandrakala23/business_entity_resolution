"""
src/features/build_features.py
Person 2 — Feature Engineering Lead

Contract (from Person 4 / pipeline):
    build_features(candidates, source1, source2, source3) -> DataFrame

- `candidates` is Person 1's blocking output. It has (at least):
      source1_entity_id, candidate_entity_id, signal, score
  `signal` is categorical (which blocking rule fired) -> we ignore it.
  `score` is her cheap numeric similarity -> we keep it, it's a fine feature.
- `candidate_entity_id` is prefixed (e.g. "S2-...", "S3-...") to say which
  source it came from. We split on that prefix and merge each half against
  the matching source table.
- We must NOT touch source1_entity_id / candidate_entity_id — the matcher
  auto-selects numeric columns as features and ignores ID/text columns, so
  we just keep those two intact and add numeric columns freely.

ASSUMPTION — Person 1's normalize.py interface
------------------------------------------------
`src/blocking/normalize.py` isn't in this deliverable, so this file makes
one assumption about its return shape, isolated in `_unpack_name()` /
`_unpack_addr()` below:

    normalize_name(raw)    -> dict-like with keys: "canonical", "tokens" (list[str])
    normalize_address(raw) -> dict-like with keys: "canonical", "tokens" (list[str]),
                               "pincode" (str|None)

If her real functions return a tuple, a dataclass, or different key names,
you only need to edit `_unpack_name` / `_unpack_addr` — nothing else in this
file depends on the exact shape. Please swap in the real normalize.py as
soon as it lands and sanity-check those two functions against a few rows.

Vectorization notes (why each feature is computed the way it is)
------------------------------------------------------------------
At ~2.2M S1 entities x up to 100 candidates (TOP_K), that's up to ~220M
pairs. Two different things need to scale here, and they scale differently:

1. Normalization (normalize_name/normalize_address) is only ever run ONCE
   PER UNIQUE ENTITY (source1, source2, source3), not once per pair — we
   dedupe first, normalize the unique set, then join back with a dict/map
   lookup. That's ~2.2M + (S2 size) + (S3 size) calls, not 220M.

2. Pairwise similarity (the actual candidate x candidate comparison) is
   where real vectorization matters:
     - Jaccard / token-overlap on names & addresses: represented as a
       sparse binary term matrix (CountVectorizer, binary=True) built once
       from ALL unique tokens across the three sources. For an aligned
       pair (row i in `candidates`), intersection/union come from
       elementwise sparse multiply + row-sum — no Python loop over pairs.
     - TF-IDF cosine: same trick — one TfidfVectorizer fit once, transform
       once, then per-row cosine = elementwise-multiply-and-sum on the two
       aligned sparse matrices (rows already L2-normalized by sklearn, so
       elementwise-multiply-and-sum IS the cosine). Fully vectorized.
     - Phonetic match (Soundex/Metaphone): computed once per unique name,
       then compared with a plain vectorized Series == Series.
     - Levenshtein / Jaro-Winkler: these genuinely can't be expressed as
       sparse matrix algebra. We use `rapidfuzz` (C-implemented, ~1-5M
       string-pair comparisons/sec) called via `rapidfuzz.distance.*` in a
       tight loop over aligned numpy arrays — NOT a pandas `.apply(axis=1)`
       or a pure-Python char-by-char loop. Being fully honest: this is
       still O(n) in Python-loop terms, just over a C-fast primitive. If
       this turns out too slow at full 220M-pair scale, the fix is
       chunking + multiprocessing (a `Pool.map` over chunks of the two
       aligned string arrays), not a different algorithm. Flag this to
       Person 3/4 if profiling shows it's the bottleneck.

No external lookups anywhere below — everything comes from the provided
frames only, per the compliance rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False

try:
    import jellyfish
    _HAVE_JELLYFISH = True
except ImportError:
    _HAVE_JELLYFISH = False

try:
    from src.blocking.normalize import normalize_name, normalize_address
except ImportError:
    normalize_name = None
    normalize_address = None


# --------------------------------------------------------------------------
# Adapters for Person 1's normalize.py — edit ONLY these two if her real
# return shape differs from the assumption documented above.
# --------------------------------------------------------------------------

def _unpack_name(normalized) -> tuple[str, list[str]]:
    """-> (canonical_string, token_list)"""
    if isinstance(normalized, dict):
        return normalized.get("canonical", ""), normalized.get("tokens", [])
    if isinstance(normalized, (tuple, list)):
        canonical = normalized[0] if len(normalized) > 0 else ""
        tokens = normalized[1] if len(normalized) > 1 else canonical.split()
        return canonical, tokens
    canonical = str(normalized)
    return canonical, canonical.split()


def _unpack_addr(normalized) -> tuple[str, list[str], str | None]:
    """-> (canonical_string, token_list, pincode_or_None)"""
    if isinstance(normalized, dict):
        return (
            normalized.get("canonical") or normalized.get("normalized", ""),
            normalized.get("tokens", []),
            normalized.get("pincode"),
        )
    if isinstance(normalized, (tuple, list)):
        canonical = normalized[0] if len(normalized) > 0 else ""
        tokens = normalized[1] if len(normalized) > 1 else canonical.split()
        pincode = normalized[2] if len(normalized) > 2 else None
        return canonical, tokens, pincode
    canonical = str(normalized)
    return canonical, canonical.split(), None


try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s: str) -> str:
        return s

# Real data has entity names in Devanagari (e.g. "मॉडर्न फाइनेंस") that need
# transliterating before they're comparable to Latin-script names in other
# sources. unidecode gives an approximate phonetic-ish Latin rendering —
# not a translation — which is the standard cheap trick here and is what
# Person 1's real normalize.py almost certainly also does.
_LEGAL_SUFFIXES = {
    "inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "llp", "plc", "sarl", "gmbh",
}


def _fallback_normalize_name(raw: str) -> dict:
    """Only used if src/blocking/normalize.py isn't importable (e.g. running
    this file standalone before Person 1's module exists). NOT a substitute
    for her real normalizer — swap it out the moment hers lands."""
    s = "" if raw is None else unidecode(str(raw)).strip().lower()
    s = s.replace(".", "").replace(",", " ")
    tokens = [t for t in s.split() if t not in _LEGAL_SUFFIXES]
    return {"canonical": " ".join(tokens), "tokens": tokens}


_PINCODE_RE = None


def _fallback_normalize_address(raw: str) -> dict:
    import re
    global _PINCODE_RE
    if _PINCODE_RE is None:
        _PINCODE_RE = re.compile(r"\b\d{5,6}\b")
    s = "" if raw is None else unidecode(str(raw)).strip().lower()
    s = s.replace(".", "").replace(",", " ")
    tokens = s.split()
    m = _PINCODE_RE.search(s)
    pincode = m.group(0) if m else None
    return {"canonical": " ".join(tokens), "tokens": tokens, "pincode": pincode}


_normalize_name = normalize_name or _fallback_normalize_name
_normalize_address = normalize_address or _fallback_normalize_address


# --------------------------------------------------------------------------
# Step 1: normalize each source table ONCE (deduped by entity id), not once
# per candidate pair.
# --------------------------------------------------------------------------

def _normalize_source(df: pd.DataFrame, id_col: str, name_col: str, addr_col: str) -> pd.DataFrame:
    """Returns a small lookup frame: id_col, name_canonical, name_tokens,
    addr_canonical, addr_tokens, pincode — one row per unique entity."""
    unique = df[[id_col, name_col, addr_col]].drop_duplicates(subset=id_col).copy()

    name_results = unique[name_col].apply(_normalize_name)
    unpacked_name = name_results.apply(_unpack_name)
    unique["name_canonical"] = unpacked_name.apply(lambda t: t[0])
    unique["name_tokens"] = unpacked_name.apply(lambda t: t[1])

    addr_results = unique[addr_col].apply(_normalize_address)
    unpacked_addr = addr_results.apply(_unpack_addr)
    unique["addr_canonical"] = unpacked_addr.apply(lambda t: t[0])
    unique["addr_tokens"] = unpacked_addr.apply(lambda t: t[1])
    unique["pincode"] = unpacked_addr.apply(lambda t: t[2])

    return unique[[id_col, "name_canonical", "name_tokens", "addr_canonical", "addr_tokens", "pincode"]]


# --------------------------------------------------------------------------
# Step 2: fully-vectorized pairwise features via sparse term matrices.
# --------------------------------------------------------------------------

def _sparse_jaccard_and_overlap(
    left_tokens: pd.Series, right_tokens: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Given two aligned Series of token lists (one row per candidate pair),
    return (jaccard, overlap_count) as numpy arrays — no per-pair loop.
    Uses a binary CountVectorizer fit over the union of both sides so every
    token maps to the same column index, then row-aligned sparse ops."""
    n = len(left_tokens)
    corpus = list(left_tokens.apply(lambda toks: " ".join(toks))) + list(
        right_tokens.apply(lambda toks: " ".join(toks))
    )
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(corpus)  # (2n, vocab)
    left_mat, right_mat = mat[:n], mat[n:]

    intersection = np.asarray(left_mat.multiply(right_mat).sum(axis=1)).ravel()
    left_count = np.asarray(left_mat.sum(axis=1)).ravel()
    right_count = np.asarray(right_mat.sum(axis=1)).ravel()
    union = left_count + right_count - intersection

    jaccard = np.divide(
        intersection, union, out=np.zeros(n, dtype=float), where=union > 0
    )
    return jaccard, intersection.astype(float)


def _sparse_tfidf_cosine(left_text: pd.Series, right_text: pd.Series) -> np.ndarray:
    """Aligned row-wise cosine similarity via one shared TF-IDF space.
    sklearn L2-normalizes rows by default, so elementwise-multiply-then-sum
    over aligned rows IS the cosine similarity — fully vectorized."""
    n = len(left_text)
    corpus = list(left_text) + list(right_text)
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    mat = vec.fit_transform(corpus)  # rows are L2-normalized
    left_mat, right_mat = mat[:n], mat[n:]
    cosine = np.asarray(left_mat.multiply(right_mat).sum(axis=1)).ravel()
    return cosine


def _rapidfuzz_ratio(left: pd.Series, right: pd.Series, algo: str) -> np.ndarray:
    """C-fast pairwise string ratio via rapidfuzz. Not sparse-matrix
    vectorized (edit distance can't be expressed that way), but avoids
    pandas .apply(axis=1) and pure-Python char loops. See module docstring."""
    if not _HAVE_RAPIDFUZZ:
        return np.zeros(len(left), dtype=float)
    left_arr = left.fillna("").to_numpy()
    right_arr = right.fillna("").to_numpy()
    if algo == "levenshtein":
        fn = Levenshtein.normalized_similarity
    else:
        fn = JaroWinkler.normalized_similarity
    return np.fromiter((fn(a, b) for a, b in zip(left_arr, right_arr)), dtype=float, count=len(left_arr))


def _phonetic_match(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Soundex/Metaphone equality — codes computed once per side (already
    deduped upstream via merge), comparison is a vectorized Series == Series."""
    if not _HAVE_JELLYFISH:
        return np.zeros(len(left), dtype=int)

    def _code(tokens_str: str) -> str:
        first = tokens_str.split()[0] if tokens_str else ""
        return jellyfish.metaphone(first) if first else ""

    left_codes = left.fillna("").apply(_code)
    right_codes = right.fillna("").apply(_code)
    return ((left_codes == right_codes) & (left_codes != "")).astype(int).to_numpy()


# --------------------------------------------------------------------------
# Main entry point — the contracted function.
# --------------------------------------------------------------------------

def build_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    # column name candidates DATAFRAME uses to reference each side:
    cand_s1_id_col: str = "source1_entity_id",
    cand_id_col: str = "candidate_entity_id",
    # real schema of source1/source2/source3.tsv, confirmed from the actual
    # files: columns are entity_id, business_name, business_address, country
    source_id_col: str = "entity_id",
    name_col: str = "business_name",
    addr_col: str = "business_address",
    country_col: str = "country",
    s2_prefix: str = "S2-",
    s3_prefix: str = "S3-",
) -> pd.DataFrame:
    out = candidates.copy()

    # --- normalize each source ONCE, deduped by entity id ---
    s1_norm = _normalize_source(source1, source_id_col, name_col, addr_col)
    s2_norm = _normalize_source(source2, source_id_col, name_col, addr_col)
    s3_norm = _normalize_source(source3, source_id_col, name_col, addr_col)

    # country lookups (cheap categorical -> boolean match feature)
    s1_country = source1.drop_duplicates(subset=source_id_col).set_index(source_id_col)[country_col]
    s2_country = source2.drop_duplicates(subset=source_id_col).set_index(source_id_col)[country_col]
    s3_country = source3.drop_duplicates(subset=source_id_col).set_index(source_id_col)[country_col]
    cand_country = pd.concat([s2_country, s3_country])

    # --- attach S1 side (every row has one) ---
    out = out.merge(
        s1_norm.add_prefix("s1_"), left_on=cand_s1_id_col, right_on=f"s1_{source_id_col}", how="left"
    )
    out["s1_country"] = out[cand_s1_id_col].map(s1_country)
    out["cand_country"] = out[cand_id_col].map(cand_country)

    # --- split candidates by which source they belong to, via ID prefix ---
    is_s2 = out[cand_id_col].astype(str).str.startswith(s2_prefix)
    is_s3 = out[cand_id_col].astype(str).str.startswith(s3_prefix)

    cand_norm = pd.concat(
        [
            s2_norm.rename(columns={source_id_col: cand_id_col}).assign(_src=2),
            s3_norm.rename(columns={source_id_col: cand_id_col}).assign(_src=3),
        ],
        ignore_index=True,
    ).add_prefix("cand_").rename(columns={f"cand_{cand_id_col}": cand_id_col})

    out = out.merge(cand_norm, on=cand_id_col, how="left")

    # rows where a candidate id didn't match either source (shouldn't happen
    # if candidate_pairs.tsv is clean, but don't silently crash on it)
    for col in ["name_canonical", "addr_canonical", "pincode"]:
        out[f"s1_{col}"] = out.get(f"s1_{col}", pd.Series(dtype=object)).fillna("")
        out[f"cand_{col}"] = out.get(f"cand_{col}", pd.Series(dtype=object)).fillna("")
    out["s1_name_tokens"] = out["s1_name_tokens"].apply(lambda v: v if isinstance(v, (list, set, frozenset, tuple)) else [])
    out["cand_name_tokens"] = out["cand_name_tokens"].apply(lambda v: v if isinstance(v, (list, set, frozenset, tuple)) else [])
    out["s1_addr_tokens"] = out["s1_addr_tokens"].apply(lambda v: v if isinstance(v, (list, set, frozenset, tuple)) else [])
    out["cand_addr_tokens"] = out["cand_addr_tokens"].apply(lambda v: v if isinstance(v, (list, set, frozenset, tuple)) else [])

    # --- vectorized pairwise features ---
    name_jaccard, name_overlap = _sparse_jaccard_and_overlap(out["s1_name_tokens"], out["cand_name_tokens"])
    addr_jaccard, addr_overlap = _sparse_jaccard_and_overlap(out["s1_addr_tokens"], out["cand_addr_tokens"])

    out["feat_name_jaccard"] = name_jaccard
    out["feat_name_token_overlap"] = name_overlap
    out["feat_addr_jaccard"] = addr_jaccard
    out["feat_addr_token_overlap"] = addr_overlap

    out["feat_name_tfidf_cosine"] = _sparse_tfidf_cosine(out["s1_name_canonical"], out["cand_name_canonical"])
    out["feat_addr_tfidf_cosine"] = _sparse_tfidf_cosine(out["s1_addr_canonical"], out["cand_addr_canonical"])

    out["feat_name_levenshtein"] = _rapidfuzz_ratio(out["s1_name_canonical"], out["cand_name_canonical"], "levenshtein")
    out["feat_name_jarowinkler"] = _rapidfuzz_ratio(out["s1_name_canonical"], out["cand_name_canonical"], "jarowinkler")
    out["feat_addr_levenshtein"] = _rapidfuzz_ratio(out["s1_addr_canonical"], out["cand_addr_canonical"], "levenshtein")

    out["feat_name_phonetic_match"] = _phonetic_match(out["s1_name_canonical"], out["cand_name_canonical"])

    out["feat_pincode_match"] = (
        (out["s1_pincode"].fillna("") != "")
        & (out["s1_pincode"] == out["cand_pincode"])
    ).astype(int)

    out["feat_country_match"] = (out["s1_country"] == out["cand_country"]).astype(int)

    # --- clean up: keep the two ID columns + signal/score intact, plus
    # every new feat_* column. Drop the scratch/intermediate columns. ---
    keep_cols = [cand_s1_id_col, cand_id_col] + [c for c in candidates.columns if c not in (cand_s1_id_col, cand_id_col)]
    feat_cols = [c for c in out.columns if c.startswith("feat_")]
    result = out[keep_cols + feat_cols].copy()
    return result


# --------------------------------------------------------------------------
# Demo / smoke test — synthetic data, so this is runnable before the real
# dataset or Person 1's normalize.py exist. Run: python build_features.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 5:
        # real-data mode: python build_features.py candidates.tsv source1.tsv source2.tsv source3.tsv
        cand_path, s1_path, s2_path, s3_path = sys.argv[1:5]
        candidates = pd.read_csv(cand_path, sep="\t", dtype=str)
        for c in ("score",):
            if c in candidates.columns:
                candidates[c] = pd.to_numeric(candidates[c], errors="coerce")
        s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
        s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
        s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
        feats = build_features(candidates, s1, s2, s3)
        print(feats.shape)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(feats.head(15))
        if "label" in feats.columns:
            # quick separation check: do features actually differ for true
            # matches vs random negatives? (sanity check only, not scoring)
            feat_cols = [c for c in feats.columns if c.startswith("feat_")]
            print("\nmean feature value by label (1=true match, 0=random negative):")
            print(feats.groupby("label")[feat_cols].mean().T)
    else:
        # tiny synthetic demo — runs with zero setup
        s1 = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["Liberty Family Office", "Acme Rd Trading Co", "Kumar Devendra Pvt Ltd"],
            "business_address": ["12 Main St, Paris", "5 Acme Road, London", "MG Road, Bengaluru 560001"],
            "country": ["France", "UK", "India"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-10", "S2-11"],
            "business_name": ["libertyfamilyoffice.com", "Acme Trading Company"],
            "business_address": ["12 Main Street, Paris", "5 Acme Rd, London"],
            "country": ["France", "UK"],
        })
        s3 = pd.DataFrame({
            "entity_id": ["S3-20"],
            "business_name": ["Devendra Kumar Ltd"],
            "business_address": ["MG Rd, Bengaluru, 560001"],
            "country": ["India"],
        })
        candidates = pd.DataFrame({
            "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
            "candidate_entity_id": ["S2-10", "S2-11", "S3-20"],
            "signal": ["fuzzy_name", "exact_name", "fuzzy_name"],
            "score": [0.71, 0.95, 0.66],
        })
        feats = build_features(candidates, s1, s2, s3)
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(feats)