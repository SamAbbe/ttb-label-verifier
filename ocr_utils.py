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


def _projection_score(img: Image.Image, angle: float) -> float:
    """Score how well `angle` aligns text lines horizontally.

    Rotates a binarized copy of the image, sums dark pixels per row, and
    returns the variance of that row-sum profile. Horizontal text lines
    produce sharp peaks/troughs (high variance); a skewed image smears
    dark pixels across rows (low variance).
    """
    rotated = img.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    arr = np.array(rotated)
    row_sums = np.sum(arr < 128, axis=1)
    return float(np.var(row_sums))


def deskew_image(img: Image.Image) -> Image.Image:
    """Straighten a slightly rotated label photo before OCR.

    Many real-world label photos are captured at a slight angle. Tesseract's
    accuracy drops sharply once text lines aren't roughly horizontal, so we
    estimate and correct that angle here using a projection-profile search:
    try a range of rotation angles and keep the one that makes the image's
    row-by-row "ink density" profile most peaked (i.e. text lines line up
    horizontally).
    """
    bw = img.point(lambda x: 0 if x < 128 else 255)

    best_angle, best_score = 0.0, _projection_score(bw, 0.0)
    for angle in np.arange(-15, 15.5, 0.5):
        score = _projection_score(bw, angle)
        if score > best_score:
            best_angle, best_score = float(angle), score

    # Refine around the coarse best angle.
    for angle in np.arange(best_angle - 0.5, best_angle + 0.5, 0.1):
        score = _projection_score(bw, angle)
        if score > best_score:
            best_angle, best_score = float(angle), score

    if abs(best_angle) < 0.1:
        return img

    return img.rotate(best_angle, expand=True, fillcolor=255, resample=Image.BICUBIC)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Light preprocessing to help OCR on real-world label photos.

    Handles the kinds of issues stakeholders called out: low resolution,
    poor lighting, glare, and slightly skewed photos. This is a cheap
    heuristic pass, not a full computer-vision pipeline -- documented as a
    limitation in the README.
    """
    img = image.convert("L")  # grayscale
    img = deskew_image(img)

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
    """Run OCR on a label image and return the best raw text result.

    Label layouts vary a lot - some are dense paragraphs, some are
    scattered text blocks around artwork. We try a couple of Tesseract
    page-segmentation modes and keep whichever produced the most text,
    which in practice is a decent proxy for "read the label correctly".
    """
    processed = preprocess_image(image)

    candidates = []
    for psm in ("11", "3"):
        try:
            text = pytesseract.image_to_string(processed, config=f"--psm {psm}")
            candidates.append(text)
        except pytesseract.TesseractNotFoundError:
            # Not an OCR-quality issue - Tesseract itself isn't installed/on
            # PATH. Let this propagate so the caller can show a clear setup
            # error instead of silently treating the label as blank.
            raise
        except Exception:
            continue

    if not candidates:
        return ""
    return max(candidates, key=len)
