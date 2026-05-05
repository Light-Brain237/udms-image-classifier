"""Fix cell 23 — use in-memory model instead of loading from disk."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

cells[23]['source'] = [
    "# EarlyStopping(restore_best_weights=True) already restored the best weights\n",
    "# into `model` — no need to reload from disk (avoids Lambda deserialization issues).\n",
    "best_model = model\n",
    "print('Using in-memory model (best weights already restored by EarlyStopping).')\n",
    "\n",
    "# Run inference on the held-out test set (15%)\n",
    "y_true_list, y_prob_list = [], []\n",
    "for images, labels in test_ds:\n",
    "    probs = best_model.predict(images, verbose=0)\n",
    "    y_prob_list.append(probs)\n",
    "    y_true_list.append(np.argmax(labels.numpy(), axis=1))\n",
    "\n",
    "y_true = np.concatenate(y_true_list)\n",
    "y_prob = np.concatenate(y_prob_list)\n",
    "y_pred = np.argmax(y_prob, axis=1)\n",
    "\n",
    "test_acc = accuracy_score(y_true, y_pred)\n",
    "print(f'Test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)')\n",
    "print(f'Test samples  : {len(y_true):,}')\n",
    "print()\n",
    "print('Classification Report:')\n",
    "print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))\n",
]

# Also fix the TFLite export cell (27) to use best_model directly (already in memory)
# Check it uses best_model — it should already be correct since we set best_model = model

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cell 23 fixed.')
