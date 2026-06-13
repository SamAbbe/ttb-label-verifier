"""AI-Powered Alcohol Label Verification - Streamlit prototype.

Lets a TTB compliance agent upload a label image and the corresponding
application data, then automatically checks whether the label matches the
application across the core required fields, including a dedicated check
for the mandatory Government Warning statement.

Run locally:
    streamlit run app.py

See README.md for setup, approach, and known limitations.
"""

import io
import time

import pandas as pd
import streamlit as st
from PIL import Image

import pytesseract

from ocr_utils import extract_text
from verification import FIELD_LABELS, verify_label, overall_status


STATUS_COLORS = {
    "MATCH": "#1a7f37",
    "PASS": "#1a7f37",
    "REVIEW": "#9a6700",
    "NEEDS REVIEW": "#9a6700",
    "MISMATCH": "#cf222e",
    "FAIL": "#cf222e",
    "SKIPPED": "#6e7781",
}

STATUS_ICONS = {
    "MATCH": "✅",
    "PASS": "✅",
    "REVIEW": "⚠️",
    "NEEDS REVIEW": "⚠️",
    "MISMATCH": "❌",
    "FAIL": "❌",
    "SKIPPED": "⏭️",
}


def status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#6e7781")
    icon = STATUS_ICONS.get(status, "")
    return (
        f'<span style="background-color:{color}; color:white; padding:3px 10px; '
        f'border-radius:12px; font-weight:600; font-size:0.9em;">{icon} {status}</span>'
    )


def results_to_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Field": r["field"],
            "Status": f'{STATUS_ICONS.get(r["status"], "")} {r["status"]}',
            "Expected (from application)": r["expected"],
            "Found on label (best match)": r["found"],
            "Confidence": r["score"],
            "Notes": r["note"],
        })
    return pd.DataFrame(rows)


def render_results(results: list, elapsed: float):
    overall = overall_status(results)
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"**Overall result:**<br>{status_badge(overall)}", unsafe_allow_html=True)
    with col2:
        st.caption(f"Processed in {elapsed:.1f} seconds")

    df = results_to_dataframe(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if overall == "FAIL":
        st.error("One or more required fields do not match the application. Recommend rejection or further review.")
    elif overall == "NEEDS REVIEW":
        st.warning("Some fields could not be confidently verified. An agent should review these manually.")
    else:
        st.success("All required fields match the application.")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Label Verification Assistant", page_icon="🍷", layout="wide")

st.title("🍷 Label Verification Assistant")
st.markdown(
    "Upload a label image and the corresponding application details. "
    "The tool checks that the label matches the application and flags "
    "the mandatory Government Warning statement automatically."
)

tab_single, tab_batch, tab_about = st.tabs(["Single Label Check", "Batch Upload", "About"])

# ---------------------------------------------------------------------------
# Single label check
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("1. Enter application data")

    col_left, col_right = st.columns(2)
    with col_left:
        brand_name = st.text_input("Brand Name", placeholder="e.g. OLD TOM DISTILLERY")
        class_type = st.text_input("Class/Type Designation", placeholder="e.g. Kentucky Straight Bourbon Whiskey")
        alcohol_content = st.text_input("Alcohol Content", placeholder="e.g. 45% Alc./Vol. (90 Proof)")
    with col_right:
        net_contents = st.text_input("Net Contents", placeholder="e.g. 750 mL")
        bottler_info = st.text_input("Name & Address of Bottler/Producer", placeholder="e.g. Old Tom Distillery, Bardstown, KY")
        country_of_origin = st.text_input("Country of Origin (imports only)", placeholder="leave blank for domestic")

    st.subheader("2. Upload label image")
    uploaded_image = st.file_uploader("Label photo or scan", type=["png", "jpg", "jpeg"], key="single_image")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded label", width=300)

    if st.button("Verify Label", type="primary", disabled=uploaded_image is None):
        application_data = {
            "brand_name": brand_name,
            "class_type": class_type,
            "alcohol_content": alcohol_content,
            "net_contents": net_contents,
            "bottler_info": bottler_info,
            "country_of_origin": country_of_origin,
        }

        start = time.time()
        image = Image.open(uploaded_image)
        try:
            ocr_text = extract_text(image)
        except pytesseract.TesseractNotFoundError:
            st.error(
                "Tesseract OCR is not installed or not on the system PATH. "
                "Verification cannot run until it's installed - see the "
                "README for setup instructions."
            )
            st.stop()
        results = verify_label(application_data, ocr_text)
        elapsed = time.time() - start

        st.subheader("3. Results")
        render_results(results, elapsed)

        with st.expander("Raw text extracted from label (OCR)"):
            st.text(ocr_text or "(no text detected)")

# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown(
        "For batch processing (e.g. a large importer submitting 200-300 "
        "applications at once): upload a CSV of application data and the "
        "matching label images in one go."
    )

    st.markdown("**CSV columns required:** `filename, brand_name, class_type, "
                 "alcohol_content, net_contents, bottler_info, country_of_origin`")
    st.caption("`filename` must match the uploaded image's filename exactly (e.g. `old_tom.png`). "
               "`country_of_origin` may be left blank for domestic products.")

    sample_csv = pd.DataFrame([{
        "filename": "old_tom.png",
        "brand_name": "OLD TOM DISTILLERY",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler_info": "Old Tom Distillery, Bardstown, KY",
        "country_of_origin": "",
    }])
    st.download_button(
        "Download sample CSV template",
        data=sample_csv.to_csv(index=False),
        file_name="sample_applications.csv",
        mime="text/csv",
    )

    csv_file = st.file_uploader("Application data (CSV)", type=["csv"], key="batch_csv")
    image_files = st.file_uploader(
        "Label images (select multiple)", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="batch_images",
    )

    if st.button("Run Batch Verification", type="primary", disabled=not (csv_file and image_files)):
        df = pd.read_csv(csv_file, dtype=str).fillna("")
        images_by_name = {f.name: f for f in image_files}

        summary_rows = []
        detail_results = {}
        progress = st.progress(0.0, text="Starting...")

        for i, row in df.iterrows():
            filename = str(row.get("filename", "")).strip()
            progress.progress((i + 1) / len(df), text=f"Processing {filename or f'row {i+1}'}...")

            if filename not in images_by_name:
                summary_rows.append({
                    "filename": filename,
                    "brand_name": row.get("brand_name", ""),
                    "overall": "ERROR",
                    "details": f"No matching image uploaded for '{filename}'",
                })
                continue

            image_file = images_by_name[filename]
            image_file.seek(0)
            image = Image.open(image_file)
            try:
                ocr_text = extract_text(image)
            except pytesseract.TesseractNotFoundError:
                progress.empty()
                st.error(
                    "Tesseract OCR is not installed or not on the system PATH. "
                    "Verification cannot run until it's installed - see the "
                    "README for setup instructions."
                )
                st.stop()

            application_data = {key: row.get(key, "") for key, _ in FIELD_LABELS}
            results = verify_label(application_data, ocr_text)
            overall = overall_status(results)

            issues = [f"{r['field']}: {r['status']}" for r in results if r["status"] in ("MISMATCH", "REVIEW")]

            summary_rows.append({
                "filename": filename,
                "brand_name": row.get("brand_name", ""),
                "overall": overall,
                "details": "; ".join(issues) if issues else "All fields match",
            })
            detail_results[filename] = results

        progress.empty()

        summary_df = pd.DataFrame(summary_rows)
        summary_df["Status"] = summary_df["overall"].map(lambda s: f'{STATUS_ICONS.get(s, "")} {s}')

        st.subheader("Batch results")
        st.dataframe(
            summary_df[["filename", "brand_name", "Status", "details"]],
            use_container_width=True, hide_index=True,
        )

        n_pass = (summary_df["overall"] == "PASS").sum()
        n_review = (summary_df["overall"] == "NEEDS REVIEW").sum()
        n_fail = (summary_df["overall"] == "FAIL").sum()
        n_error = (summary_df["overall"] == "ERROR").sum()
        st.caption(f"{n_pass} passed · {n_review} need review · {n_fail} failed · {n_error} errors")

        st.download_button(
            "Download results as CSV",
            data=summary_df[["filename", "brand_name", "overall", "details"]].to_csv(index=False),
            file_name="batch_verification_results.csv",
            mime="text/csv",
        )

        for filename, results in detail_results.items():
            with st.expander(f"Details: {filename}"):
                st.dataframe(results_to_dataframe(results), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown("""
### How it works

1. **OCR extraction** - Tesseract reads all visible text from the uploaded label image.
2. **Field comparison** - each application field (brand name, class/type, ABV, net
   contents, bottler info, country of origin) is fuzzy-matched against the extracted
   text, so minor formatting differences (e.g. "STONE'S THROW" vs "Stone's Throw")
   are not flagged as errors.
3. **Government Warning check** - the mandatory health warning statement gets a
   dedicated check: the wording must match the statement required by 27 CFR 16.21,
   and "GOVERNMENT WARNING:" must appear in capital letters.
4. **Result** - each field is marked **MATCH**, **REVIEW** (possible match, needs a
   human look), or **MISMATCH**, rolling up to an overall PASS / NEEDS REVIEW / FAIL.

### Why this approach

- Runs entirely locally (Tesseract OCR + Python text matching) - no API keys, no
  outbound network calls, which matters given the agency's firewall restrictions
  noted in the discovery interviews.
- Designed to be fast (well under the 5-second threshold the team called out) and
  the interface is intentionally minimal - one screen, one button, plain-language
  results.
- Batch upload supports the "200-300 applications at once" peak-season scenario.

See the project README for full documentation of assumptions, trade-offs, and
suggested next steps for a production version.
""")
