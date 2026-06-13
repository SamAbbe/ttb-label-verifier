# Label Verification Assistant

A prototype tool that helps TTB compliance agents check whether an alcohol
beverage label matches its COLA application - automatically verifying brand
name, class/type, alcohol content, net contents, bottler info, country of
origin, and the mandatory Government Warning statement.

## Why this approach

Based on the discovery notes, the prototype was designed around four
constraints that mattered more than raw model sophistication:

- **Speed**: the scanning-vendor pilot failed because it took 30-40 seconds
  per label. This tool runs OCR + comparison locally and returns results in
  a few seconds.
- **Network restrictions**: the agency firewall blocks many outbound
  domains, including ML API endpoints. Everything here runs locally
  (Tesseract OCR + Python text matching) - no API keys, no internet
  connection required at runtime.
- **Usability for non-technical staff**: a single page, one button, plain
  pass/review/fail results with explanations - aimed at the "my mother could
  figure it out" bar from the interview notes.
- **Judgment, not just pattern matching**: fields are fuzzy-matched so that
  "STONE'S THROW" vs "Stone's Throw" isn't flagged as an error, while the
  Government Warning gets a stricter, dedicated check because its wording
  and formatting requirements are legally exact.

## Tools used

- **Python 3.10**
- **Streamlit** - UI framework (chosen for speed of development and a
  built-in clean, accessible interface with no frontend code needed)
- **Tesseract OCR** (via `pytesseract`) - local, offline text extraction
- **Pillow** - image preprocessing (grayscale, contrast boost, upscaling,
  sharpening) to improve OCR accuracy on imperfect photos
- **pandas** - CSV handling for batch uploads
- Python's built-in `difflib` for fuzzy text matching (no extra ML
  dependency required)

## Setup & run instructions

### 1. Install Tesseract OCR (one-time, system-level)

- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: install from
  https://github.com/UB-Mannheim/tesseract/wiki and ensure `tesseract.exe`
  is on your PATH (or set `pytesseract.pytesseract.tesseract_cmd` in
  `ocr_utils.py` to its install path).

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) generate sample test labels

```bash
cd sample_data
python generate_samples.py
cd ..
```

This creates five synthetic label images covering different scenarios
(perfect match, formatting differences, a bad Government Warning, a
mismatched ABV, and a slightly skewed import label) plus
`sample_applications.csv` with the matching application data for batch
testing.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Using the app

**Single Label Check** - enter the application's field values, upload a
label image, click "Verify Label". Each field shows MATCH / REVIEW /
MISMATCH with a confidence score and an explanation.

**Batch Upload** - for large importers submitting many applications at once:
upload a CSV (template available in-app) plus all the label images at once.
Each row is matched to an image by filename and processed in sequence; results
can be downloaded as a CSV.

## Assumptions & scope

- This is a **standalone prototype**, not integrated with COLA, per the
  project notes ("not looking to integrate with COLA directly").
- No PII or application data is stored - everything is processed in-memory
  for the duration of the session.
- The Government Warning is checked against the standard statement required
  by 27 CFR 16.21. (Some product categories have alternate/abbreviated
  wording rules; this prototype checks against the standard full statement.)
- "Country of Origin" is treated as optional (left blank for domestic
  products), per the requirement that it only applies to imports.
- Field matching uses normalized fuzzy text comparison (case-insensitive,
  punctuation-tolerant), with a stricter numeric check for fields like ABV
  and net contents so that, e.g., "40%" vs "35%" is always flagged even
  though the surrounding text is similar.

## Known limitations & trade-offs

- **OCR accuracy on poor-quality photos**: basic preprocessing (contrast
  boost, upscaling, sharpening) helps, but Tesseract will still struggle
  with extreme angles, heavy glare, or very low resolution. A production
  version could add perspective correction or a vision-LLM fallback for
  low-confidence cases.
- **Bold/formatting of "GOVERNMENT WARNING:"**: Tesseract does not report
  font weight, and case detection from OCR is not 100% reliable. The
  prototype checks for a literal all-caps "GOVERNMENT WARNING:" string, and
  falls back to a "needs review" flag rather than silently passing when this
  can't be confirmed. A production version would likely need layout/format
  analysis (e.g., bounding box + font metrics) for a fully automated check.
- **No persistence / audit log**: results aren't saved. A production version
  would need an audit trail per federal record-retention requirements.
- **Single style of label layout assumed in samples**: the matching logic
  itself doesn't assume a fixed layout (it searches the full OCR text), but
  the sample labels are simple and computer-generated rather than real
  photographs.

## Project structure

```
label_verifier/
├── app.py                  # Streamlit UI (single + batch)
├── ocr_utils.py             # OCR extraction & image preprocessing
├── verification.py          # Field comparison & Government Warning logic
├── requirements.txt
├── packages.txt              # system packages for Streamlit Cloud (tesseract-ocr)
└── sample_data/
    ├── generate_samples.py   # creates synthetic test labels
    └── sample_applications.csv
```
