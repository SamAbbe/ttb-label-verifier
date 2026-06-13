"""OCR helpers for extracting text from photographed/scanned alcohol labels.

Uses Tesseract (via pytesseract) - runs fully locally, no API key required,
no outbound network calls. That matters for the Treasury network constraints
described in the project notes.
"""

import os
import shutil

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pytesseract

# On Windows, Tesseract is often installed but not added to PATH. Fall back
# to the default install location so pytesseract can find the binary without
# requiring users to edit their PATH manually.
if shutil.which("tesseract") is None:
    default_windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(default_windows_path):
        pytesseract.pytesseract.tesseract_cmd = default_windows_path


def preprocess_image(image: Image.Image) -> Image.Image:
    """Light preprocessing to help OCR on real-world label photos.

    Handles the kinds of issues stakeholders called out: low resolution,
    poor lighting, glare, and slightly skewed photos. This is a cheap
    heuristic pass, not a full computer-vision pipeline -- documented as a
    limitation in the README.

    Note: an earlier version of this function attempted to auto-deskew
    (straighten) rotated images using a row-projection-profile search.
    That heuristic turned out to be unreliable on these label designs - the
    bottle/border shapes and gradient backgrounds could dominate the
    projection profile and cause it to "correct" already-straight images
    into heavily rotated (and far less readable) ones. It was removed in
    favor of the simpler, more predictable pipeline below. Handling
    skewed/rotated photos robustly is called out as future work in the
    README.
    """
    img = image.convert("L")  # grayscale

    # Light text on a dark background (common on dark bottle labels) reads
    # very poorly with Tesseract's defaults, which expect dark text on a
    # light background. Invert if the image is predominantly dark.
    if np.array(img).mean() < 128:
        img = ImageOps.invert(img)

    # Upscale small images - Tesseract is much more accurate around
    # ~1500px on the long edge.
    if max(img.size) < 1500:
        scale = 1500 / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Auto-contrast helps with low-light photos and mild glare.
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)

    return img


def extract_text(image: Image.Image) -> str:
    """Run OCR on a label image and return the combined raw text result.

    Label layouts vary a lot - some are dense paragraphs, some are
    scattered text blocks around artwork. We try a couple of Tesseract
    page-segmentation modes and combine their output.

    Note: this used to pick whichever single mode produced the longest
    text, on the theory that "more text" roughly meant "read the label
    correctly". In practice that's version-dependent - on some Tesseract
    builds, "--psm 11" (sparse text, no particular order) can return a
    longer but much noisier/garbled result than "--psm 3" (uniform block),
    which made the picked text unreliable across environments (e.g. local
    vs. Streamlit Cloud, which may ship a different Tesseract version).
    Concatenating both is more robust: the verification logic searches for
    the best-matching substring, so extra noise from one mode doesn't hurt
    a clean match found in the other.
    """
    processed = preprocess_image(image)

    candidates = []
    for psm in ("11", "3"):
        try:
            text = pytesseract.image_to_string(processed, config=f"--psm {psm}")
            text = text.strip()
            if text:
                candidates.append(text)
        except pytesseract.TesseractNotFoundError:
            # Not an OCR-quality issue - Tesseract itself isn't installed/on
            # PATH. Let this propagate so the caller can show a clear setup
            # error instead of silently treating the label as blank.
            raise
        except Exception:
            continue

    # De-duplicate identical results (common when both PSM modes agree).
    seen = []
    for text in candidates:
        if text not in seen:
            seen.append(text)

    return "\n".join(seen)
