"""Core verification logic: compares application data against OCR'd label text.

Design notes (see README for more):
- Uses normalized fuzzy text matching (stdlib difflib) so that formatting
  differences like "STONE'S THROW" vs "Stone's Throw" are treated as a
  match, per Dave Morrison's feedback in the discovery notes.
- The Government Warning gets dedicated handling because the requirement
  is stricter: exact wording, AND "GOVERNMENT WARNING:" must appear in
  all caps.
- Numeric values (ABV, proof, net contents) are checked separately from
  the general fuzzy match, since "40%" and "35%" look textually similar
  but mean very different things.
- Every result includes a confidence score and a human-readable note so
  agents can quickly see *why* something passed/failed/needs review,
  rather than just a flat pass/fail.
"""

import difflib
import re

# Standard Alcohol Beverage Health Warning Statement required by
# 27 CFR 16.21. Wording must match exactly; "GOVERNMENT WARNING:" must be
# bold and in capital letters.
STANDARD_GOV_WARNING = (
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD "
    "NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF "
    "BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR "
    "ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH "
    "PROBLEMS."
)

MATCH_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.55

# Application-data field -> human-readable label shown in the UI/report
FIELD_LABELS = [
    ("brand_name", "Brand Name"),
    ("class_type", "Class/Type Designation"),
    ("alcohol_content", "Alcohol Content"),
    ("net_contents", "Net Contents"),
    ("bottler_info", "Name & Address of Bottler/Producer"),
    ("country_of_origin", "Country of Origin"),
]


def normalize(text: str) -> str:
    """Uppercase, strip punctuation noise, collapse whitespace.

    This is what makes "Stone's Throw" and "STONE'S THROW." compare equal,
    and absorbs common OCR artifacts (stray periods, extra spaces, etc.)
    """
    text = text.upper()
    text = re.sub(r"[^A-Z0-9.%/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def best_match_ratio(needle: str, haystack: str):
    """Return (similarity ratio 0-1, best matching substring of haystack).

    First checks for a direct substring match (fast path, ratio = 1.0).
    Otherwise slides a window of roughly-needle-sized length over the
    haystack and returns the best SequenceMatcher ratio found. This lets
    us locate a short expected value (e.g. a brand name) within a large
    blob of OCR'd label text.
    """
    needle_n = normalize(needle)
    haystack_n = normalize(haystack)

    if not needle_n:
        return 0.0, ""

    if needle_n in haystack_n:
        return 1.0, needle_n

    words = haystack_n.split()
    needle_word_count = max(len(needle_n.split()), 1)

    if not words:
        return 0.0, ""

    best_ratio = 0.0
    best_window = ""
    # allow the window to be a bit longer/shorter than the needle
    for window_size in (needle_word_count, needle_word_count + 2, max(needle_word_count - 1, 1)):
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i + window_size])
            ratio = difflib.SequenceMatcher(None, needle_n, window).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_window = window

    return best_ratio, best_window


def verify_field(field_label: str, expected_value, ocr_text: str) -> dict:
    """Compare one expected application-data value against the OCR text."""
    expected_value = "" if expected_value is None else str(expected_value).strip()

    if not expected_value:
        return {
            "field": field_label,
            "expected": "",
            "found": "",
            "status": "SKIPPED",
            "score": None,
            "note": "No value provided in application data",
        }

    ratio, matched_text = best_match_ratio(expected_value, ocr_text)

    if ratio >= MATCH_THRESHOLD:
        status = "MATCH"
        note = "Exact match" if ratio >= 0.999 else "Matches (minor formatting differences only)"
    elif ratio >= REVIEW_THRESHOLD:
        status = "REVIEW"
        note = "Possible match - please verify manually"
    else:
        status = "MISMATCH"
        note = "Not found on label, or does not match the application"

    # Numbers (ABV, proof, volume, etc.) need to match exactly even if the
    # surrounding text is similar - "40% Alc./Vol." and "35% Alc./Vol." score
    # high on plain text similarity but mean very different things on a
    # label. If the expected value contains digits and the best-matching
    # text on the label contains different digits, treat it as a mismatch
    # regardless of the overall text similarity score.
    if ratio < 0.999:
        expected_numbers = re.findall(r"\d+(?:\.\d+)?", normalize(expected_value))
        found_numbers = re.findall(r"\d+(?:\.\d+)?", matched_text)
        if expected_numbers and expected_numbers != found_numbers:
            status = "MISMATCH"
            note = (
                f"Numeric value differs: application says "
                f"{'/'.join(expected_numbers)}, label shows "
                f"{'/'.join(found_numbers) if found_numbers else 'no matching number'}"
            )

    return {
        "field": field_label,
        "expected": expected_value,
        "found": matched_text,
        "status": status,
        "score": round(ratio, 2),
        "note": note,
    }


def verify_alcohol_content_format(ocr_text: str) -> dict:
    """Check the label's alcohol content statement against 27 CFR 5.65.

    Per the TTB mandatory labeling checklist:
      - Alcohol content must be stated as a percentage by volume, using
        an acceptable abbreviation ("Alc.", "Alc", "Vol.", "Vol", or "%").
      - A proof statement is optional, but if present it must be adequately
        distinguished from the percentage-by-volume statement (e.g. shown
        in parentheses or brackets).
    """
    field = "Alcohol Content Format (27 CFR 5.65)"
    upper_ocr = ocr_text.upper()

    pct_match = re.search(r"\d+(?:\.\d+)?\s*%", upper_ocr)
    if not pct_match:
        return {
            "field": field,
            "expected": "Alcohol content stated as a percentage by volume (e.g. 'Alc. 40% by Vol.')",
            "found": "",
            "status": "MISMATCH",
            "score": None,
            "note": "No percentage alcohol-by-volume statement found on the label - "
                    "alcohol content must be stated as a percentage by volume (27 CFR 5.65).",
        }

    # Look for an acceptable "Alc"/"Vol" abbreviation near the percentage.
    window_start = max(pct_match.start() - 20, 0)
    window_end = min(pct_match.end() + 20, len(upper_ocr))
    window = upper_ocr[window_start:window_end].strip()
    has_alc_or_vol = "ALC" in window or "VOL" in window

    notes = []
    status = "MATCH"
    if not has_alc_or_vol:
        status = "REVIEW"
        notes.append(
            "A percentage was found but no 'Alc'/'Vol' abbreviation nearby - "
            "verify the alcohol content statement uses an acceptable format "
            "(e.g. 'Alc. __% by Vol.')"
        )
    else:
        notes.append("Alcohol content is stated as a percentage by volume with an acceptable abbreviation")

    # An optional proof statement must be set apart from the % ABV statement,
    # e.g. shown in parentheses or brackets.
    proof_match = re.search(r"[(\[]?\s*\d+(?:\.\d+)?\s*PROOF\s*[)\]]?", upper_ocr)
    if proof_match:
        proof_text = proof_match.group()
        enclosed = ("(" in proof_text and ")" in proof_text) or ("[" in proof_text and "]" in proof_text)
        if not enclosed:
            status = "REVIEW" if status == "MATCH" else status
            notes.append(
                "A proof statement was found but does not appear to be enclosed in "
                "parentheses/brackets - a proof statement must be adequately "
                "distinguished from the mandatory % alcohol-by-volume statement (27 CFR 5.65)"
            )

    return {
        "field": field,
        "expected": "% Alc./Vol. (with an optional proof statement in parentheses/brackets)",
        "found": window,
        "status": status,
        "score": None,
        "note": " | ".join(notes),
    }


def verify_government_warning(ocr_text: str) -> dict:
    """Dedicated check for the mandatory Government Warning statement.

    Per 27 CFR Part 16 and the TTB mandatory labeling checklist, all of the
    following must be true:
      1. The wording matches the standard statement exactly.
      2. "GOVERNMENT WARNING" appears in capital letters (and bold, though
         bold cannot be checked via OCR).
      3. "Surgeon" and "General" are capitalized ("Surgeon General").

    Caveat (documented in README): OCR case detection is not fully
    reliable, and Tesseract does not report bold/font-weight at all, so
    the "bold" requirement cannot be verified automatically. When wording
    matches but a capitalization check is inconclusive, we mark the result
    as REVIEW rather than a silent pass.
    """
    upper_ocr = ocr_text.upper()
    warning_start = upper_ocr.find("GOVERNMENT WARNING")

    if warning_start == -1:
        return {
            "field": "Government Warning Statement",
            "expected": STANDARD_GOV_WARNING,
            "found": "",
            "status": "MISMATCH",
            "score": 0.0,
            "note": "Government Warning not found on label. This statement is mandatory on all alcohol beverages.",
        }

    # Compare only from "GOVERNMENT WARNING" onward, so unrelated text earlier
    # on the label (net contents, ABV, bottler info, etc.) doesn't get pulled
    # into the comparison and dilute the similarity score.
    warning_section = ocr_text[warning_start:]
    expected_n = normalize(STANDARD_GOV_WARNING)
    matched_text = normalize(warning_section)
    # autojunk=False: the default autojunk heuristic treats frequently-
    # repeated characters as "junk" once a sequence is >=200 chars, which
    # tanks the ratio for this statement even when the wording matches
    # almost exactly.
    ratio = difflib.SequenceMatcher(None, expected_n, matched_text, autojunk=False).ratio()
    has_allcaps_prefix = "GOVERNMENT WARNING:" in ocr_text  # case-sensitive check

    notes = []
    if ratio >= MATCH_THRESHOLD:
        notes.append("Warning text matches the required statement")
        status = "MATCH"
    elif ratio >= REVIEW_THRESHOLD:
        notes.append("Warning text differs from the required wording - verify manually")
        status = "REVIEW"
    else:
        notes.append("Warning text does not match the required statement")
        status = "MISMATCH"

    if not has_allcaps_prefix and status == "MATCH":
        status = "REVIEW"
        notes.append(
            "Could not confirm 'GOVERNMENT WARNING:' is rendered in all caps and bold - "
            "OCR case detection is unreliable, please verify visually (required by regulation)"
        )

    # "Surgeon" and "General" must be capitalized.
    sg_match = re.search(r"surgeon\s+general", warning_section, re.IGNORECASE)
    if sg_match:
        surgeon_word, general_word = sg_match.group().split()
        if not (surgeon_word[0] == "S" and general_word[0] == "G"):
            if status == "MATCH":
                status = "REVIEW"
            notes.append(
                "'Surgeon' and 'General' do not appear to be capitalized - "
                "the 'S' in Surgeon and 'G' in General must be capitalized (27 CFR Part 16)"
            )

    return {
        "field": "Government Warning Statement",
        "expected": STANDARD_GOV_WARNING,
        "found": matched_text,
        "status": status,
        "score": round(ratio, 2),
        "note": " | ".join(notes),
    }


def verify_label(application_data: dict, ocr_text: str) -> list:
    """Run every field check for one label and return a list of result dicts."""
    results = []
    for field_key, field_label in FIELD_LABELS:
        results.append(verify_field(field_label, application_data.get(field_key, ""), ocr_text))
    results.append(verify_alcohol_content_format(ocr_text))
    results.append(verify_government_warning(ocr_text))
    return results


def overall_status(results: list) -> str:
    """Roll per-field results up into one overall status for the application."""
    statuses = [r["status"] for r in results if r["status"] != "SKIPPED"]
    if "MISMATCH" in statuses:
        return "FAIL"
    if "REVIEW" in statuses:
        return "NEEDS REVIEW"
    return "PASS"
