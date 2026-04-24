"""Fix all UDMS_CATEGORIES index-ordering bugs in colab_train.ipynb."""
import json

with open("notebooks/colab_train.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

fixes = 0

# ── Cell 10: class distribution chart + sample preview ───────────────────────
src = "".join(nb["cells"][9]["source"])

old = "bars = ax.barh(UDMS_CATEGORIES, class_counts, color=colors)"
new = "bars = ax.barh(class_names, class_counts, color=colors)"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

old = "for cat, cnt in zip(UDMS_CATEGORIES, class_counts):"
new = "for cat, cnt in zip(class_names, class_counts):"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

old = "    cat = UDMS_CATEGORIES[int(np.argmax(lbl))]"
new = "    cat = class_names[int(np.argmax(lbl))]"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

nb["cells"][9]["source"] = src
print(f"Cell 10: 3 fixes applied")

# ── Cell 18: classification report ───────────────────────────────────────────
src = "".join(nb["cells"][17]["source"])
old = "print(classification_report(y_true, y_pred, target_names=UDMS_CATEGORIES, digits=3))"
new = "print(classification_report(y_true, y_pred, target_names=class_names, digits=3))"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1
nb["cells"][17]["source"] = src
print(f"Cell 18: 1 fix applied")

# ── Cell 19: confusion matrix tick labels ────────────────────────────────────
src = "".join(nb["cells"][18]["source"])
old = "ax.set_xticklabels(UDMS_CATEGORIES, rotation=45, ha='right', fontsize=9)"
new = "ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

old = "ax.set_yticklabels(UDMS_CATEGORIES, fontsize=9)"
new = "ax.set_yticklabels(class_names, fontsize=9)"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

nb["cells"][18]["source"] = src
print(f"Cell 19: 2 fixes applied")

# ── Cell 20: per-class confidence + misclassification pairs ──────────────────
src = "".join(nb["cells"][19]["source"])

# enumerate(zip(..., UDMS_CATEGORIES))
old = "for idx, (ax, cat) in enumerate(zip(axes.flat, UDMS_CATEGORIES)):"
new = "for idx, (ax, cat) in enumerate(zip(axes.flat, class_names)):"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

# off_diag misclassification pairs
old = "off_diag = [(cm[i, j], UDMS_CATEGORIES[i], UDMS_CATEGORIES[j])"
new = "off_diag = [(cm[i, j], class_names[i], class_names[j])"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

nb["cells"][19]["source"] = src
print(f"Cell 20: 2 fixes applied")

# ── Cell 23: smoke-test class name lookup ────────────────────────────────────
src = "".join(nb["cells"][22]["source"])

old = "true_class  = UDMS_CATEGORIES[int(np.argmax(sample_lbls[0]))]"
new = "true_class  = class_names[int(np.argmax(sample_lbls[0]))]"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

old = "pred_class = UDMS_CATEGORIES[pred_idx]"
new = "pred_class = class_names[pred_idx]"
assert old in src, f"Not found: {old}"
src = src.replace(old, new); fixes += 1

nb["cells"][22]["source"] = src
print(f"Cell 23: 2 fixes applied")

print(f"\nTotal fixes: {fixes}")

with open("notebooks/colab_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
