"""Generates synthetic alcohol label images for testing the verification app.

These are simple programmatically-drawn labels (not photos), but each one
is built to exercise a specific scenario described in the take-home
instructions:

  label_01_match.png      - everything matches the application exactly
  label_02_case_diff.png  - brand name case differs (label vs application) -
                             should still MATCH (fuzzy match)
  label_03_bad_warning.png- "Government Warning" in title case, not ALL CAPS -
                             should be flagged for REVIEW
  label_04_wrong_abv.png  - ABV on label differs from the application -
                             should MISMATCH
  label_05_import_skewed.png - import product, slightly rotated/skewed photo
                             to test OCR robustness on imperfect images

Run:
    python generate_samples.py

Requires only Pillow (already in requirements.txt). Fonts are bundled in
./fonts so this works the same on Windows/macOS/Linux.
"""

from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

STANDARD_WARNING_LINES = [
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD",
    "NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF",
    "BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR",
    "ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH",
    "PROBLEMS.",
]

BAD_WARNING_LINES = [
    "Government Warning: (1) According to the Surgeon General, women should",
    "not drink alcoholic beverages during pregnancy because of the risk of",
    "birth defects. (2) Consumption of alcoholic beverages impairs your",
    "ability to drive a car or operate machinery, and may cause health",
    "problems.",
]

# Fonts are bundled alongside this script so the generator works the same
# on Windows/macOS/Linux without depending on system fonts being installed.
FONT_DIR = os.path.join(OUT_DIR, "fonts")
FONT_BOLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
FONT_REGULAR = os.path.join(FONT_DIR, "DejaVuSans.ttf")


def draw_label(filename, brand, class_type, abv, net_contents, bottler,
               country_of_origin=None, warning_lines=None, rotate=0):
    width, height = 900, 1100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_brand = ImageFont.truetype(FONT_BOLD, 54)
    font_section = ImageFont.truetype(FONT_BOLD, 30)
    font_body = ImageFont.truetype(FONT_REGULAR, 28)
    font_small = ImageFont.truetype(FONT_REGULAR, 20)
    font_warning_label = ImageFont.truetype(FONT_BOLD, 20)

    y = 60
    draw.rectangle([20, 20, width - 20, height - 20], outline="black", width=3)

    y += 40
    draw.text((width // 2, y), brand, font=font_brand, fill="black", anchor="mm")
    y += 80
    draw.line([(80, y), (width - 80, y)], fill="black", width=2)

    y += 50
    draw.text((width // 2, y), class_type, font=font_section, fill="black", anchor="mm")

    y += 70
    draw.text((width // 2, y), f"{abv}", font=font_body, fill="black", anchor="mm")

    y += 60
    draw.text((width // 2, y), f"Net Contents: {net_contents}", font=font_body, fill="black", anchor="mm")

    y += 60
    draw.text((width // 2, y), bottler, font=font_small, fill="black", anchor="mm")

    if country_of_origin:
        y += 40
        draw.text((width // 2, y), f"Product of {country_of_origin}", font=font_small, fill="black", anchor="mm")

    # Government warning block, bottom of label
    if warning_lines:
        y = height - 220
        draw.line([(80, y), (width - 80, y)], fill="black", width=1)
        y += 25
        for line in warning_lines:
            draw.text((60, y), line, font=font_warning_label if "WARNING" in line.upper() else font_small,
                      fill="black")
            y += 28

    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor="white")

    img.save(os.path.join(OUT_DIR, filename))
    print(f"wrote {filename}")


def main():
    # 1. Everything matches
    draw_label(
        "label_01_match.png",
        brand="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        abv="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
        bottler="Old Tom Distillery, Bardstown, KY",
        warning_lines=STANDARD_WARNING_LINES,
    )

    # 2. Brand name case differs from application (label uses title case)
    draw_label(
        "label_02_case_diff.png",
        brand="Stone's Throw",
        class_type="American Dry Gin",
        abv="40% Alc./Vol. (80 Proof)",
        net_contents="750 mL",
        bottler="Stone's Throw Distilling Co., Seattle, WA",
        warning_lines=STANDARD_WARNING_LINES,
    )

    # 3. Government warning not in ALL CAPS / wrong formatting
    draw_label(
        "label_03_bad_warning.png",
        brand="HARBOR LIGHT BREWING",
        class_type="India Pale Ale",
        abv="6.5% Alc./Vol.",
        net_contents="12 FL OZ",
        bottler="Harbor Light Brewing Co., Portland, OR",
        warning_lines=BAD_WARNING_LINES,
    )

    # 4. ABV on label doesn't match application (application says 40%)
    draw_label(
        "label_04_wrong_abv.png",
        brand="SILVER CREEK VODKA",
        class_type="Vodka",
        abv="35% Alc./Vol. (70 Proof)",
        net_contents="1 L",
        bottler="Silver Creek Distillers, Denver, CO",
        warning_lines=STANDARD_WARNING_LINES,
    )

    # 5. Import product, slightly rotated to simulate an imperfect photo
    draw_label(
        "label_05_import_skewed.png",
        brand="VALLE VERDE",
        class_type="Blanco Tequila",
        abv="40% Alc./Vol. (80 Proof)",
        net_contents="750 mL",
        bottler="Destileria Valle Verde, Jalisco, Mexico",
        country_of_origin="Mexico",
        warning_lines=STANDARD_WARNING_LINES,
        rotate=3,
    )


if __name__ == "__main__":
    main()
