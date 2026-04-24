"""Deep audit: find every cell that uses UDMS_CATEGORIES as an index-ordered list."""
import json

nb = json.load(open("notebooks/colab_train.ipynb", encoding="utf-8"))

# These are patterns that indicate UDMS_CATEGORIES is used as index->name mapping
# (i.e. bugs), not just as a definition or for CATEGORY_LABELS lookup
BAD_PATTERNS = [
    "UDMS_CATEGORIES[",          # direct index lookup
    "zip(UDMS_CATEGORIES",       # zipping with counts/indices
    "enumerate(UDMS_CATEGORIES", # enumerating
    "target_names=UDMS_CATEGORIES",  # classification report
    "barh(UDMS_CATEGORIES",      # chart labels
    "set_xticklabels(UDMS_CATEGORIES",
    "set_yticklabels(UDMS_CATEGORIES",
]

print("Cells with UDMS_CATEGORIES used as index-ordered list (BUGS):")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    found = [p for p in BAD_PATTERNS if p in src]
    if found:
        print(f"\n  Cell {i+1}:")
        for p in found:
            # Show the line containing the pattern
            for line in src.splitlines():
                if p in line:
                    print(f"    LINE: {line.strip()}")

print()
print("scripts/test_tflite.py label_map loading:")
tflite_src = open("scripts/test_tflite.py", encoding="utf-8").read()
for line in tflite_src.splitlines():
    if "label" in line.lower():
        print(f"  {line}")
