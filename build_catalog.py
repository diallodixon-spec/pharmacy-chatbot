"""
Regenerates the embedded product catalog inside api/index.py from a
product CSV export.

Run this any time the SuperMed Pharmacy product sheet changes:

    python3 build_catalog.py path/to/products.csv

Expected CSV columns (matches the current export from the site):
    product, brand, category, product_type, package_size,
    active_ingredient_or_key_ingredient, price_jmd, price_numeric_jmd,
    shop_page, source_url, data_quality_note

Only the fields the chatbot actually needs are kept, to minimize the
token count of the catalog that gets sent to OpenAI on every request.

The catalog is embedded directly inside api/index.py as a JSON literal
(PRODUCTS = json.loads(r'''...''')) rather than imported from a
separate module, because Vercel's Python bundler does not reliably
resolve plain sibling-file imports for the function entrypoint. This
script finds that block and replaces it in place.
"""

import csv
import json
import re
import sys

INDEX_PATH = "api/index.py"


def build_catalog(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    catalog = []
    for row in rows:
        name = (row.get("product") or "").strip()
        if not name:
            continue

        brand = (row.get("brand") or "").strip()
        if brand.lower() in ("generic / not stated", "not stated", ""):
            brand = ""

        ingredient = (row.get("active_ingredient_or_key_ingredient") or "").strip()
        if ingredient.lower().startswith("not identifiable"):
            ingredient = ""

        catalog.append(
            {
                "name": name,
                "brand": brand,
                "category": (row.get("category") or "").strip(),
                "ingredient": ingredient,
                "price_jmd": (row.get("price_jmd") or "").strip(),
                "url": "https://supermedpharmacy.com/shop/",
            }
        )

    return catalog


def splice_into_index(catalog: list[dict]) -> int:
    with open(INDEX_PATH, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"PRODUCTS = json\.loads\(r'''\n.*?\n'''\)", re.S)
    if not pattern.search(content):
        print(
            f"Could not find the PRODUCTS block in {INDEX_PATH}. "
            "Has the file structure changed? Aborting without writing."
        )
        sys.exit(1)

    replacement = "PRODUCTS = json.loads(r'''\n" + json.dumps(catalog, ensure_ascii=False) + "\n''')"
    new_content = pattern.sub(replacement, content, count=1)

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    return len(catalog)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 build_catalog.py path/to/products.csv")
        sys.exit(1)

    catalog = build_catalog(sys.argv[1])
    count = splice_into_index(catalog)
    print(f"Updated {INDEX_PATH} with {count} products.")


if __name__ == "__main__":
    main()
