"""
Normalization utilities for business_name and business_address fields.

Design notes:
- No external lookups (API calls, geocoding, gov databases) are used anywhere here.
  `unidecode` is a local, offline transliteration *library* (rule-based Unicode
  table), not an external service — it never looks up a specific business.
- Everything here is deterministic and re-runnable on train or test alike, and
  is written to be agnostic to country: it does not branch on country == 'US'/
  'India' in a way that would break on an unseen country like France. State/city
  extraction only *helps* when a match is found in a small static list of
  well-known US/India region names; it degrades gracefully to "unknown" for
  everything else (France included) rather than failing.
"""
import re
from unidecode import unidecode

# --- Legal-suffix / stopword tokens to strip from business names -----------
# These inflate block sizes without carrying entity-distinguishing signal.
LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "pvt", "private", "plc",
    "gmbh", "sarl", "sa", "ag", "bv", "pty", "enterprises", "enterprise",
    "group", "holdings", "solutions", "services", "consultants",
    "consulting", "associates", "international", "global", "the",
}

# --- Address abbreviation normalization (bidirectional -> canonical form) --
ADDR_ABBR = {
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bdr\b": "drive", r"\bln\b": "lane", r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment", r"\bste\b": "suite", r"\bhwy\b": "highway",
    r"\bpl\b": "place", r"\bct\b": "court", r"\bcir\b": "circle",
    r"\bpkwy\b": "parkway", r"\bter\b": "terrace",
}

# Small static geography helpers — bundled reference lists (not live lookups).
# Used only to *assist* blocking; absence just means we fall back to token-only.
US_STATE_ABBR = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy",
}

PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")
PINCODE_RE = re.compile(r"\b(\d{5,6})\b")

# Generic address-structure words that carry no entity-distinguishing signal
# (kept separate from US_STATE_ABBR / city names, which we WANT as blocking keys).
ADDR_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "apartment",
    "suite", "highway", "place", "court", "circle", "parkway", "terrace",
    "floor", "near", "behind", "off", "house", "flat", "plot", "block",
    "wing", "village", "taluka", "dist", "district", "ward", "sector",
    "phase", "stage", "no", "number", "survey", "gali",
    "city", "town",
}


def has_non_latin(text: str) -> bool:
    return bool(re.search(r"[^\x00-\x7F]", text))


def normalize_name(raw: str):
    """
    Returns dict with:
      canonical      - sorted, suffix-stripped, space-joined token string
                        (order-invariant, used as primary blocking key material)
      tokens         - set of significant tokens (post suffix-strip)
      first_token    - first significant token (secondary key)
      was_transliterated - bool, True if unidecode was applied
    """
    if raw is None:
        raw = ""
    text = raw.strip()
    transliterated = False
    if has_non_latin(text):
        text = unidecode(text)
        transliterated = True

    text = text.lower()
    text = text.replace("&", " and ")
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()

    tokens = [t for t in text.split(" ") if t]
    sig_tokens = [t for t in tokens if t not in LEGAL_SUFFIX_TOKENS and len(t) > 1]
    if not sig_tokens:
        # fall back to full token list if stripping removed everything
        # (e.g. name IS just "Global Solutions")
        sig_tokens = tokens

    canonical = " ".join(sorted(sig_tokens))
    first_token = sig_tokens[0] if sig_tokens else ""
    # concatenated, no-space, sorted-token key: catches "Liberty Family Office"
    # vs "libertyfamilyoffice.com" style domain-name variants
    concat_key = "".join(sig_tokens)
    prefix3 = first_token[:3] if len(first_token) >= 3 else ""

    return {
        "canonical": canonical,
        "tokens": frozenset(sig_tokens),
        "first_token": first_token,
        "concat_key": concat_key,
        "prefix3": prefix3,
        "was_transliterated": transliterated,
    }


def normalize_address(raw: str):
    """
    Returns dict with:
      normalized  - cleaned, abbreviation-expanded address string
      pincode     - 5-6 digit code if found, else ""
      state_abbr  - 2-letter US state token if found in the address, else ""
      tokens      - set of significant address tokens (for fallback token blocking)
    """
    if raw is None:
        raw = ""
    text = raw.strip()
    if has_non_latin(text):
        text = unidecode(text)
    text = text.lower()

    pincode_match = PINCODE_RE.search(text)
    pincode = pincode_match.group(1) if pincode_match else ""

    text_clean = PUNCT_RE.sub(" ", text)
    text_clean = WS_RE.sub(" ", text_clean).strip()

    for pattern, repl in ADDR_ABBR.items():
        text_clean = re.sub(pattern, repl, text_clean)

    tokens = [
        t for t in text_clean.split(" ")
        if len(t) > 2 and not t.isdigit() and t not in ADDR_STOPWORDS
    ]

    state_abbr = ""
    raw_tokens = text_clean.split(" ")
    for t in raw_tokens:
        if t in US_STATE_ABBR:
            state_abbr = t
            break

    return {
        "normalized": text_clean,
        "pincode": pincode,
        "state_abbr": state_abbr,
        "tokens": frozenset(tokens),
    }
