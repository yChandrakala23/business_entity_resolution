"""
Candidate generation (blocking) for the Business Entity Resolution challenge.

Strategy: inverted-index blocking via pandas joins (not naive O(n*m) pairwise
comparison). Three independent blocking signals are unioned so that a pair only
needs to agree on ONE of them to become a candidate:

  1. name_prefix : country + first-2-sorted-significant-tokens of the name
                    (word-order invariant, catches minor tail typos)
  2. name_token   : country + a single significant name token
                    (catches cases where most tokens differ/are noisy but one
                    distinctive word survives - "Cao Assets" vs "Cao Aofianehs")
                    Tokens that are too common (appear in > MAX_TOKEN_BLOCK
                    records) are dropped from this index - they're not
                    discriminative and would blow up block size for nothing.
  3. addr_pincode : country + pincode extracted from the address
                    (catches cases where the name is heavily corrupted/
                    transliterated but the address pincode survives)

Candidates are unioned, deduplicated, scored with a cheap similarity function,
and capped to top-K per Source-1 entity so the downstream matching model isn't
fed an unbounded candidate list.

No external data/APIs are used - only the provided source files.
"""
import argparse
import sys
import time
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, ".")
from normalize import normalize_name, normalize_address  # noqa: E402

RAREST_K_NAME = 3        # index each record under its N rarest name tokens
RAREST_K_ADDR = 4      # index each record under its N rarest address tokens
MAX_TOKEN_BLOCK = 5000   # a token this common (in its source) is dropped entirely - non-discriminative
TOP_K = 100              # max candidates kept per Source-1 entity


def load_source(path: str, tag: str) -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"[{tag}] loaded {len(df):,} rows in {time.time()-t0:.1f}s", file=sys.stderr)
    return df


def add_normalized_columns(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    t0 = time.time()
    name_info = df["business_name"].apply(normalize_name)
    addr_info = df["business_address"].apply(normalize_address)

    df["name_canonical"] = [d["canonical"] for d in name_info]
    df["name_tokens"] = [d["tokens"] for d in name_info]
    df["name_prefix_key"] = [
        " ".join(sorted(d["tokens"])[:2]) if d["tokens"] else ""
        for d in name_info
    ]
    df["name_concat_key"] = [d["concat_key"] for d in name_info]
    df["name_char_prefix"] = [d["prefix3"] for d in name_info]
    df["pincode"] = [d["pincode"] for d in addr_info]
    df["addr_tokens"] = [d["tokens"] for d in addr_info]
    df["addr_normalized"] = [d["normalized"] for d in addr_info]

    print(f"[{tag}] normalized in {time.time()-t0:.1f}s", file=sys.stderr)
    return df


def explode_tokens(df: pd.DataFrame, id_col: str, token_col: str) -> pd.DataFrame:
    """Long-format (entity_id, token) table for join-based blocking."""
    tmp = df[[id_col, token_col]].explode(token_col)
    tmp = tmp[tmp[token_col].notna() & (tmp[token_col] != "")]
    tmp.columns = [id_col, "token"]
    return tmp


def build_token_index_filtered(df: pd.DataFrame, id_col: str, token_col: str = "name_tokens",
                                rarest_k: int = 3) -> pd.DataFrame:
    """
    Token table indexed by each record's RAREST_K tokens only (by in-source
    document frequency), not every token under a flat threshold. This is the
    standard fix for inverted-index blowup: common tokens like "star" or
    "investment" still exist in the index, but a record is only *filed* under
    its most distinctive tokens, so block sizes stay controlled even on
    millions of rows. Tokens more common than MAX_TOKEN_BLOCK are dropped
    outright first (truly non-discriminative, e.g. leftover filler words).
    """
    long_df = explode_tokens(df, id_col, token_col)
    counts = long_df["token"].value_counts()
    long_df = long_df.merge(counts.rename("freq"), left_on="token", right_index=True)
    long_df = long_df[long_df["freq"] <= MAX_TOKEN_BLOCK]
    long_df = long_df.sort_values("freq")
    kept = long_df.groupby(id_col).head(rarest_k)
    return kept.drop(columns="freq")


def generate_candidates_for_source(s1: pd.DataFrame, s2_or_3: pd.DataFrame, src_tag: str) -> pd.DataFrame:
    """
    Returns long DataFrame: [source1_entity_id, candidate_entity_id, signal]
    where signal in {name_prefix, name_token, addr_pincode} records which
    blocking rule produced the pair (a pair can appear more than once, dedup later).
    """
    frames = []

    # --- Signal 1: name prefix (country + first 2 sorted tokens) ---
    s1_key = s1[s1["name_prefix_key"] != ""][["entity_id", "country", "name_prefix_key"]]
    s2_key = s2_or_3[s2_or_3["name_prefix_key"] != ""][["entity_id", "country", "name_prefix_key"]]
    m = s1_key.merge(s2_key, on=["country", "name_prefix_key"], suffixes=("_s1", "_cand"))
    if len(m):
        out = m[["entity_id_s1", "entity_id_cand"]].copy()
        out.columns = ["source1_entity_id", "candidate_entity_id"]
        out["signal"] = "name_prefix"
        frames.append(out)
    print(f"  [{src_tag}] name_prefix pairs: {len(m):,}", file=sys.stderr)
        # --- Signal 1b: character prefix blocking ---
    # Helps typo cases:
    # Network -> Netw0rk

    s1_char = s1[s1["name_char_prefix"] != ""][[
        "entity_id",
        "country",
        "name_char_prefix"
    ]]

    s2_char = s2_or_3[s2_or_3["name_char_prefix"] != ""][[
        "entity_id",
        "country",
        "name_char_prefix"
    ]]

    m_char = s1_char.merge(
        s2_char,
        on=["country", "name_char_prefix"],
        suffixes=("_s1", "_cand")
    )

    if len(m_char):
        out_char = m_char[
            ["entity_id_s1", "entity_id_cand"]
        ].copy()

        out_char.columns = [
            "source1_entity_id",
            "candidate_entity_id"
        ]

        out_char["signal"] = "name_char_prefix"

        frames.append(out_char)

    print(
        f"  [{src_tag}] name_char_prefix pairs: {len(m_char):,}",
        file=sys.stderr
    )
    # --- Signal 2: shared significant name token (rarest-K per record) ---
    s1_tok = build_token_index_filtered(s1, "entity_id", "name_tokens", RAREST_K_NAME)
    s1_tok = s1_tok.merge(s1[["entity_id", "country"]], on="entity_id")
    s2_tok = build_token_index_filtered(s2_or_3, "entity_id", "name_tokens", RAREST_K_NAME)
    s2_tok = s2_tok.merge(s2_or_3[["entity_id", "country"]], on="entity_id")
    m2 = s1_tok.merge(s2_tok, on=["country", "token"], suffixes=("_s1", "_cand"))
    if len(m2):
        out2 = m2[["entity_id_s1", "entity_id_cand"]].copy()
        out2.columns = ["source1_entity_id", "candidate_entity_id"]
        out2["signal"] = "name_token"
        frames.append(out2)
    print(f"  [{src_tag}] name_token pairs: {len(m2):,}", file=sys.stderr)

    # --- Signal 3: address pincode ---
    s1_pin = s1[s1["pincode"] != ""][["entity_id", "country", "pincode"]]
    s2_pin = s2_or_3[s2_or_3["pincode"] != ""][["entity_id", "country", "pincode"]]
    m3 = s1_pin.merge(s2_pin, on=["country", "pincode"], suffixes=("_s1", "_cand"))
    if len(m3):
        out3 = m3[["entity_id_s1", "entity_id_cand"]].copy()
        out3.columns = ["source1_entity_id", "candidate_entity_id"]
        out3["signal"] = "addr_pincode"
        frames.append(out3)
    print(f"  [{src_tag}] addr_pincode pairs: {len(m3):,}", file=sys.stderr)

    # --- Signal 4: shared significant address token ---
    # Crucial fallback for non-Latin-script names (S1 English vs S2/S3
    # transliterated script): addresses in this dataset stay in Latin script
    # even when the name doesn't, so address tokens still line up.
    s1_atok = build_token_index_filtered(s1, "entity_id", "addr_tokens", RAREST_K_ADDR)
    s1_atok = s1_atok.merge(s1[["entity_id", "country"]], on="entity_id")
    s2_atok = build_token_index_filtered(s2_or_3, "entity_id", "addr_tokens", RAREST_K_ADDR)
    s2_atok = s2_atok.merge(s2_or_3[["entity_id", "country"]], on="entity_id")
    m4 = s1_atok.merge(s2_atok, on=["country", "token"], suffixes=("_s1", "_cand"))
    if len(m4):
        out4 = m4[["entity_id_s1", "entity_id_cand"]].copy()
        out4.columns = ["source1_entity_id", "candidate_entity_id"]
        out4["signal"] = "addr_token"
        frames.append(out4)
    print(f"  [{src_tag}] addr_token pairs: {len(m4):,}", file=sys.stderr)

    # --- Signal 5: concatenated no-space name (domain-style variants) ---
    s1_ck = s1[s1["name_concat_key"] != ""][["entity_id", "country", "name_concat_key"]]
    s2_ck = s2_or_3[s2_or_3["name_concat_key"] != ""][["entity_id", "country", "name_concat_key"]]
    m5 = s1_ck.merge(s2_ck, on=["country", "name_concat_key"], suffixes=("_s1", "_cand"))
    if len(m5):
        out5 = m5[["entity_id_s1", "entity_id_cand"]].copy()
        out5.columns = ["source1_entity_id", "candidate_entity_id"]
        out5["signal"] = "name_concat"
        frames.append(out5)
    print(f"  [{src_tag}] name_concat pairs: {len(m5):,}", file=sys.stderr)

    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "signal"])
    return pd.concat(frames, ignore_index=True)


def score_and_cap(candidates: pd.DataFrame, s1: pd.DataFrame, cand_src: pd.DataFrame) -> pd.DataFrame:
    """
    Cheap similarity score to rank + cap candidates per S1 entity so we don't
    hand the modeling stage an unbounded list. Uses rapidfuzz token_sort_ratio
    on canonical names (cheap, vectorizable via apply) plus a small bonus for
    address token overlap.
    """
    name_lookup_s1 = s1.set_index("entity_id")["name_canonical"].to_dict()
    name_lookup_cand = cand_src.set_index("entity_id")["name_canonical"].to_dict()
    addr_lookup_s1 = s1.set_index("entity_id")["addr_tokens"].to_dict()
    addr_lookup_cand = cand_src.set_index("entity_id")["addr_tokens"].to_dict()

    candidates = candidates.drop_duplicates(["source1_entity_id", "candidate_entity_id"]).copy()

    # Plain-Python loop over pre-extracted lists: DataFrame.apply's overhead is
    # per-row Series construction, not the scoring call itself - this is the
    # same rapidfuzz call, ~100x faster just by avoiding .apply.
    s1_ids = candidates["source1_entity_id"].tolist()
    cand_ids = candidates["candidate_entity_id"].tolist()
    scores = [0.0] * len(s1_ids)
    for i in range(len(s1_ids)):
        sid, cid = s1_ids[i], cand_ids[i]
        n1 = name_lookup_s1.get(sid, "")
        n2 = name_lookup_cand.get(cid, "")
        name_score = fuzz.token_sort_ratio(n1, n2) if n1 and n2 else 0.0

        a1 = addr_lookup_s1.get(sid) or frozenset()
        a2 = addr_lookup_cand.get(cid) or frozenset()
        if a1 and a2:
            addr_overlap = len(a1 & a2) / max(1, len(a1 | a2)) * 100
        else:
            addr_overlap = 0.0
        scores[i] = 0.7 * name_score + 0.3 * addr_overlap

    candidates["score"] = scores
    candidates = candidates.sort_values(
        ["source1_entity_id", "score"], ascending=[True, False]
    )
    capped = candidates.groupby("source1_entity_id").head(TOP_K)
    return capped


def run(s1_path, s2_path, s3_path, out_path, limit=None, country=None):
    s1 = load_source(s1_path, "S1")
    s2 = load_source(s2_path, "S2")
    s3 = load_source(s3_path, "S3")

    if country:
        s1 = s1[s1["country"] == country]
        s2 = s2[s2["country"] == country]
        s3 = s3[s3["country"] == country]
        print(f"Filtered to country={country}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}", file=sys.stderr)

    if limit:
        s1 = s1.head(limit)

    s1 = add_normalized_columns(s1, "S1")
    s2 = add_normalized_columns(s2, "S2")
    s3 = add_normalized_columns(s3, "S3")

    print("Generating candidates against Source 2...", file=sys.stderr)
    cand2 = generate_candidates_for_source(s1, s2, "S2")
    print("Generating candidates against Source 3...", file=sys.stderr)
    cand3 = generate_candidates_for_source(s1, s3, "S3")

    all_cand = pd.concat([cand2, cand3], ignore_index=True)
    print(f"Total raw candidate pairs (pre-dedup): {len(all_cand):,}", file=sys.stderr)

    print("Scoring and capping to top-K per S1 entity...", file=sys.stderr)
    cand_src_combined = pd.concat([s2, s3], ignore_index=True)
    final = score_and_cap(all_cand, s1, cand_src_combined)

    # Collapse to required output shape: one row per S1 entity, comma-joined IDs
    grouped = (
        final.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedup, preserve order
        .reset_index()
    )
    grouped.columns = ["source1_entity_id", "candidate_entity_ids"]

    # Ensure every S1 entity has a row, even with zero candidates
    all_s1 = s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
    result = all_s1.merge(grouped, on="source1_entity_id", how="left")
    result["candidate_entity_ids"] = result["candidate_entity_ids"].fillna("")

    result.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(result):,} rows to {out_path}", file=sys.stderr)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="limit S1 rows (for quick testing)")
    ap.add_argument("--country", type=str, default=None,
                     help="process only this country (run once per country, then concatenate outputs - "
                          "matches never cross countries, so this bounds memory/runtime losslessly)")
    args = ap.parse_args()
    run(args.source1, args.source2, args.source3, args.out, args.limit, args.country)
