"""Generates synthetic alcohol label images for testing the verification app.

These are simple programmatically-drawn labels (not photos), built from a
small catalog of products. Each product is run through the same five
scenarios, for 10 products x 5 scenarios = 50 label images:

  match        - everything on the label matches the application exactly
  case_diff    - brand name case differs (label vs application) -
                  should still MATCH (fuzzy match)
  bad_warning  - "Government Warning" in title case, not ALL CAPS -
                  should be flagged FAIL
  wrong_abv    - ABV on label differs from the application - should MISMATCH
  skewed       - slightly rotated/skewed photo (and, where the product is an
                  import, includes a country-of-origin statement) to test
                  OCR robustness on imperfect images

Run:
    python generate_samples.py

Writes label_NN_<product>_<scenario>.png for NN 01-50, plus
sample_applications.csv with the matching application data for batch testing.

Requires only Pillow (already in requirements.txt). Fonts are bundled in
./fonts so this works the same on Windows/macOS/Linux.
"""

from PIL import Image, ImageDraw, ImageFont
import csv
import os
import numpy as np

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

# A few different font pairings so labels don't all look identical. Each
# product picks one of these (regular, bold) pairs.
FONT_SETS = {
    "sans": (FONT_REGULAR, FONT_BOLD),
    "serif": (os.path.join(FONT_DIR, "georgia.ttf"), os.path.join(FONT_DIR, "georgiab.ttf")),
    "trebuchet": (os.path.join(FONT_DIR, "trebuc.ttf"), os.path.join(FONT_DIR, "trebucbd.ttf")),
    "verdana": (os.path.join(FONT_DIR, "verdana.ttf"), os.path.join(FONT_DIR, "verdanab.ttf")),
}

# Small catalog of products. Each is run through every scenario below.
PRODUCTS = [
    {
        "slug": "old_tom",
        "brand": "OLD TOM DISTILLERY",
        "brand_title": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "abv": "45% Alc./Vol. (90 Proof)",
        "wrong_abv": "40% Alc./Vol. (80 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery, Bardstown, KY",
        "country": None,
        "bottle_color": (91, 58, 30),
        "cap_color": (40, 40, 40),
        "label_bg": (35, 25, 18),
        "border_color": (201, 162, 39),
        "text_color": (245, 240, 228),
        "font_set": "serif",
    },
    {
        "slug": "stones_throw",
        "brand": "STONE'S THROW",
        "brand_title": "Stone's Throw",
        "class_type": "American Dry Gin",
        "abv": "40% Alc./Vol. (80 Proof)",
        "wrong_abv": "35% Alc./Vol. (70 Proof)",
        "net_contents": "750 mL",
        "bottler": "Stone's Throw Distilling Co., Seattle, WA",
        "country": None,
        "bottle_color": (169, 201, 208),
        "cap_color": (27, 42, 74),
        "label_bg": (255, 255, 255),
        "border_color": (27, 42, 74),
        "text_color": (20, 20, 30),
        "font_set": "verdana",
    },
    {
        "slug": "harbor_light",
        "brand": "HARBOR LIGHT BREWING",
        "brand_title": "Harbor Light Brewing",
        "class_type": "India Pale Ale",
        "abv": "6.5% Alc./Vol.",
        "wrong_abv": "5.0% Alc./Vol.",
        "net_contents": "12 FL OZ",
        "bottler": "Harbor Light Brewing Co., Portland, OR",
        "country": None,
        "bottle_color": (138, 90, 43),
        "cap_color": (201, 162, 39),
        "label_bg": (255, 248, 231),
        "border_color": (138, 90, 43),
        "text_color": (20, 20, 20),
        "font_set": "sans",
    },
    {
        "slug": "silver_creek",
        "brand": "SILVER CREEK VODKA",
        "brand_title": "Silver Creek Vodka",
        "class_type": "Vodka",
        "abv": "40% Alc./Vol. (80 Proof)",
        "wrong_abv": "35% Alc./Vol. (70 Proof)",
        "net_contents": "1 L",
        "bottler": "Silver Creek Distillers, Denver, CO",
        "country": None,
        "bottle_color": (207, 231, 236),
        "cap_color": (141, 153, 174),
        "label_bg": (255, 255, 255),
        "border_color": (58, 71, 80),
        "text_color": (20, 20, 20),
        "font_set": "serif",
    },
    {
        "slug": "valle_verde",
        "brand": "VALLE VERDE",
        "brand_title": "Valle Verde",
        "class_type": "Blanco Tequila",
        "abv": "40% Alc./Vol. (80 Proof)",
        "wrong_abv": "35% Alc./Vol. (70 Proof)",
        "net_contents": "750 mL",
        "bottler": "Destileria Valle Verde, Jalisco, Mexico",
        "country": "Mexico",
        "bottle_color": (79, 121, 66),
        "cap_color": (179, 58, 58),
        "label_bg": (45, 33, 20),
        "border_color": (179, 58, 58),
        "text_color": (243, 233, 213),
        "font_set": "trebuchet",
    },
    {
        "slug": "red_barn",
        "brand": "RED BARN FARMS",
        "brand_title": "Red Barn Farms",
        "class_type": "Apple Brandy",
        "abv": "40% Alc./Vol. (80 Proof)",
        "wrong_abv": "35% Alc./Vol. (70 Proof)",
        "net_contents": "750 mL",
        "bottler": "Red Barn Farms Distillery, Asheville, NC",
        "country": None,
        "bottle_color": (156, 90, 46),
        "cap_color": (94, 33, 41),
        "label_bg": (251, 243, 231),
        "border_color": (94, 33, 41),
        "text_color": (20, 20, 20),
        "font_set": "verdana",
    },
    {
        "slug": "northwind",
        "brand": "NORTHWIND DISTILLERY",
        "brand_title": "Northwind Distillery",
        "class_type": "Straight Rye Whiskey",
        "abv": "45% Alc./Vol. (90 Proof)",
        "wrong_abv": "50% Alc./Vol. (100 Proof)",
        "net_contents": "750 mL",
        "bottler": "Northwind Distillery, Spokane, WA",
        "country": None,
        "bottle_color": (47, 79, 79),
        "cap_color": (184, 115, 51),
        "label_bg": (28, 28, 28),
        "border_color": (184, 115, 51),
        "text_color": (240, 240, 235),
        "font_set": "sans",
    },
    {
        "slug": "crystal_springs",
        "brand": "CRYSTAL SPRINGS",
        "brand_title": "Crystal Springs",
        "class_type": "White Rum",
        "abv": "40% Alc./Vol. (80 Proof)",
        "wrong_abv": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Crystal Springs Distillers, Austin, TX",
        "country": None,
        "bottle_color": (223, 240, 234),
        "cap_color": (42, 157, 143),
        "label_bg": (255, 255, 255),
        "border_color": (42, 157, 143),
        "text_color": (20, 20, 20),
        "font_set": "trebuchet",
    },
    {
        "slug": "highland_mist",
        "brand": "HIGHLAND MIST",
        "brand_title": "Highland Mist",
        "class_type": "Blended Scotch Whisky",
        "abv": "43% Alc./Vol. (86 Proof)",
        "wrong_abv": "40% Alc./Vol. (80 Proof)",
        "net_contents": "750 mL",
        "bottler": "Highland Mist Distillers, Speyside, Scotland",
        "country": "Scotland",
        "bottle_color": (27, 58, 43),
        "cap_color": (202, 162, 68),
        "label_bg": (22, 35, 28),
        "border_color": (202, 162, 68),
        "text_color": (240, 235, 215),
        "font_set": "verdana",
    },
    {
        "slug": "sunny_acres",
        "brand": "SUNNY ACRES",
        "brand_title": "Sunny Acres",
        "class_type": "Coffee Liqueur",
        "abv": "24% Alc./Vol.",
        "wrong_abv": "20% Alc./Vol.",
        "net_contents": "750 mL",
        "bottler": "Sunny Acres Distillery, Sacramento, CA",
        "country": None,
        "bottle_color": (78, 52, 46),
        "cap_color": (215, 168, 110),
        "label_bg": (253, 241, 227),
        "border_color": (78, 52, 46),
        "text_color": (20, 20, 20),
        "font_set": "sans",
    },
]

SCENARIOS = ["match", "case_diff", "bad_warning", "wrong_abv", "skewed"]


def _make_background(width, height, top_color=(232, 232, 236), bottom_color=(196, 198, 204)):
    """A soft vertical-gradient backdrop, like a product photo on a table."""
    top = np.array(top_color, dtype=float)
    bottom = np.array(bottom_color, dtype=float)
    rows = np.linspace(0.0, 1.0, height)[:, None]
    grad = top[None, :] * (1 - rows) + bottom[None, :] * rows
    arr = np.repeat(grad[:, None, :], width, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def draw_label(filename, brand, class_type, abv, net_contents, bottler,
               country_of_origin=None, warning_lines=None, rotate=0,
               bottle_color=(120, 120, 120), cap_color=(40, 40, 40),
               label_bg=(255, 255, 255), border_color=(40, 40, 40),
               text_color=(20, 20, 20), font_set="sans"):
    width, height = 900, 1250
    label_top = 170
    label_bottom = height - 20

    img = _make_background(width, height)
    draw = ImageDraw.Draw(img)

    font_regular_path, font_bold_path = FONT_SETS[font_set]
    font_brand = ImageFont.truetype(font_bold_path, 54)
    font_section = ImageFont.truetype(font_bold_path, 30)
    font_body = ImageFont.truetype(font_regular_path, 28)
    font_small = ImageFont.truetype(font_regular_path, 20)
    font_warning_label = ImageFont.truetype(font_bold_path, 20)

    # Bottle neck + cap, drawn behind/above the label.
    draw.rectangle([380, 40, 520, label_top + 10], fill=bottle_color)
    draw.rounded_rectangle([355, 0, 545, 55], radius=10, fill=cap_color)

    # Bottle body / label patch.
    draw.rounded_rectangle([20, label_top, width - 20, label_bottom], radius=24,
                            fill=label_bg, outline=border_color, width=6)

    y = label_top + 40

    y += 40
    draw.text((width // 2, y), brand, font=font_brand, fill=text_color, anchor="mm")
    y += 80
    draw.line([(80, y), (width - 80, y)], fill=border_color, width=2)

    y += 50
    draw.text((width // 2, y), class_type, font=font_section, fill=text_color, anchor="mm")

    y += 70
    draw.text((width // 2, y), f"{abv}", font=font_body, fill=text_color, anchor="mm")

    y += 60
    draw.text((width // 2, y), f"Net Contents: {net_contents}", font=font_body, fill=text_color, anchor="mm")

    y += 60
    draw.text((width // 2, y), bottler, font=font_small, fill=text_color, anchor="mm")

    if country_of_origin:
        y += 40
        draw.text((width // 2, y), f"Product of {country_of_origin}", font=font_small, fill=text_color, anchor="mm")

    # Government warning block, bottom of label
    if warning_lines:
        y = label_bottom - 220
        draw.line([(80, y), (width - 80, y)], fill=border_color, width=1)
        y += 25
        for line in warning_lines:
            draw.text((60, y), line, font=font_warning_label if "WARNING" in line.upper() else font_small,
                      fill=text_color)
            y += 28

    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=(214, 215, 220))

    img.save(os.path.join(OUT_DIR, filename))
    print(f"wrote {filename}")


def main():
    rows = []
    n = 0
    for product in PRODUCTS:
        for scenario in SCENARIOS:
            n += 1
            filename = f"label_{n:02d}_{product['slug']}_{scenario}.png"

            label_brand = product["brand"]
            label_abv = product["abv"]
            warning_lines = STANDARD_WARNING_LINES
            country = product["country"]
            rotate = 0

            if scenario == "case_diff":
                label_brand = product["brand_title"]
            elif scenario == "bad_warning":
                warning_lines = BAD_WARNING_LINES
            elif scenario == "wrong_abv":
                label_abv = product["wrong_abv"]
            elif scenario == "skewed":
                rotate = 3

            draw_label(
                filename,
                brand=label_brand,
                class_type=product["class_type"],
                abv=label_abv,
                net_contents=product["net_contents"],
                bottler=product["bottler"],
                country_of_origin=country,
                warning_lines=warning_lines,
                rotate=rotate,
                bottle_color=product["bottle_color"],
                cap_color=product["cap_color"],
                label_bg=product["label_bg"],
                border_color=product["border_color"],
                text_color=product["text_color"],
                font_set=product["font_set"],
            )

            rows.append({
                "filename": filename,
                "brand_name": product["brand"],
                "class_type": product["class_type"],
                "alcohol_content": product["abv"],
                "net_contents": product["net_contents"],
                "bottler_info": product["bottler"],
                "country_of_origin": country or "",
            })

    csv_path = os.path.join(OUT_DIR, "sample_applications.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "brand_name", "class_type", "alcohol_content",
            "net_contents", "bottler_info", "country_of_origin",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
