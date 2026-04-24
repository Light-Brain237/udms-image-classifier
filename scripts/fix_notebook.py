"""Patch colab_train.ipynb: fix label_map ordering + add class weights."""
import json

with open("notebooks/colab_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Fix 1: Cell 8 — compute class weights after datasets are loaded ───────────
cell8_src = "".join(nb["cells"][8]["source"])
OLD_8 = "print(f'Batch shape: {next(iter(train_ds))[0].shape}')"
NEW_8 = (
    "print(f'Batch shape: {next(iter(train_ds))[0].shape}')\n"
    "\n"
    "# ── Compute class weights to counter imbalance ───────────────────────────────\n"
    "class_names = train_ds.class_names  # alphabetical, matches model output indices\n"
    "# Count samples per class by iterating the training dataset once\n"
    "_label_counts = [0] * len(class_names)\n"
    "for _, y_batch in train_ds:\n"
    "    for y in y_batch.numpy():\n"
    "        _label_counts[y.argmax()] += 1\n"
    "_total = sum(_label_counts)\n"
    "class_weight = {i: _total / (len(class_names) * cnt) for i, cnt in enumerate(_label_counts)}\n"
    "print('Class weights (inverse-frequency):')\n"
    "for i, name in enumerate(class_names):\n"
    "    print(f'  {i} {name:<22} count={_label_counts[i]:>5}  weight={class_weight[i]:.3f}')"
)
assert OLD_8 in cell8_src, f"Snippet not found in cell 8:\n{cell8_src[-300:]}"
nb["cells"][8]["source"] = cell8_src.replace(OLD_8, NEW_8)
print("Cell 8 patched: class weight computation added")

# ── Fix 2: Cell 12 — add class_weight to Phase 1 model.fit ───────────────────
cell12_src = "".join(nb["cells"][12]["source"])
OLD_12 = (
    "history1 = model.fit(\n"
    "    train_ds,\n"
    "    validation_data=val_ds,\n"
    "    epochs=PHASE1_EPOCHS,\n"
    "    callbacks=phase1_callbacks,\n"
    ")"
)
NEW_12 = (
    "history1 = model.fit(\n"
    "    train_ds,\n"
    "    validation_data=val_ds,\n"
    "    epochs=PHASE1_EPOCHS,\n"
    "    callbacks=phase1_callbacks,\n"
    "    class_weight=class_weight,\n"
    ")"
)
assert OLD_12 in cell12_src, f"Phase 1 fit not found in cell 12"
nb["cells"][12]["source"] = cell12_src.replace(OLD_12, NEW_12)
print("Cell 12 patched: class_weight added to Phase 1 model.fit")

# ── Fix 3: Cell 14 — add class_weight to Phase 2 model.fit ───────────────────
cell14_src = "".join(nb["cells"][14]["source"])
OLD_14 = (
    "history2 = model.fit(\n"
    "    train_ds,\n"
    "    validation_data=val_ds,\n"
    "    epochs=PHASE2_EPOCHS,\n"
    "    callbacks=phase2_callbacks,\n"
    ")"
)
NEW_14 = (
    "history2 = model.fit(\n"
    "    train_ds,\n"
    "    validation_data=val_ds,\n"
    "    epochs=PHASE2_EPOCHS,\n"
    "    callbacks=phase2_callbacks,\n"
    "    class_weight=class_weight,\n"
    ")"
)
assert OLD_14 in cell14_src, f"Phase 2 fit not found in cell 14"
nb["cells"][14]["source"] = cell14_src.replace(OLD_14, NEW_14)
print("Cell 14 patched: class_weight added to Phase 2 model.fit")

# ── Fix 4: Cell 21 — use train_ds.class_names for label_map ordering ─────────
cell21_src = "".join(nb["cells"][21]["source"])
OLD_21 = (
    "# ── Write label_map.json ──────────────────────────────────────────────────────\n"
    "label_map = {\n"
    "    str(i): {'category': cat, 'label': CATEGORY_LABELS[cat]}\n"
    "    for i, cat in enumerate(UDMS_CATEGORIES)\n"
    "}\n"
)
NEW_21 = (
    "# ── Write label_map.json ──────────────────────────────────────────────────────\n"
    "# IMPORTANT: use train_ds.class_names (alphabetical order), NOT UDMS_CATEGORIES.\n"
    "# image_dataset_from_directory assigns indices alphabetically, so the label map\n"
    "# must match that order or every prediction will be mapped to the wrong class.\n"
    "label_map = {\n"
    "    str(i): {'category': cat, 'label': CATEGORY_LABELS[cat]}\n"
    "    for i, cat in enumerate(train_ds.class_names)\n"
    "}\n"
)
assert OLD_21 in cell21_src, f"label_map snippet not found in cell 21"
nb["cells"][21]["source"] = cell21_src.replace(OLD_21, NEW_21)
print("Cell 21 patched: label_map now uses train_ds.class_names")

with open("notebooks/colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nNotebook saved successfully.")
