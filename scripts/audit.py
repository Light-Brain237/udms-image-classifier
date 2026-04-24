"""Self-evaluation audit of all modified files."""
import json

print("=== models/label_map.json ===")
lm = json.load(open("models/label_map.json", encoding="utf-8"))
for k, v in lm.items():
    print(f"  {k}: {v['category']}")

print()
print("=== Notebook cell audit ===")
nb = json.load(open("notebooks/colab_train.ipynb", encoding="utf-8"))

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    flags = []
    if "class_weight" in src:
        flags.append("HAS class_weight")
    if "label_map" in src and "enumerate(UDMS_CATEGORIES)" in src:
        flags.append("BUG: label_map uses UDMS_CATEGORIES")
    if "train_ds.class_names" in src and "label_map" in src:
        flags.append("GOOD: label_map uses train_ds.class_names")
    if flags:
        print(f"  Cell {i+1}: {flags}")

print()
print("=== class_weight: compute vs use order ===")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "class_weight = {" in src:
        print(f"  COMPUTED in cell {i+1}")
    if "class_weight=class_weight" in src:
        print(f"  USED     in cell {i+1}")

print()
print("=== label_map export cell (cell 22) full source ===")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "label_map" in src and "TFLite" in src and "write_text" in src:
        print(f"  --- Cell {i+1} ---")
        print(src)

print()
print("=== class distribution chart cell: uses class_names or UDMS_CATEGORIES? ===")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "class_counts" in src and "barh" in src:
        print(f"  Cell {i+1}: uses UDMS_CATEGORIES for labels = {'UDMS_CATEGORIES' in src}")
        print(f"  Cell {i+1}: uses class_names for labels = {'class_names' in src and 'barh' in src}")

print()
print("=== smoke-test cell: uses UDMS_CATEGORIES[pred_idx] for label lookup? ===")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "smoke-test" in src and "pred_class" in src:
        print(f"  Cell {i+1} snippet:")
        idx = src.find("pred_class")
        print("  " + src[max(0, idx-100):idx+200])
