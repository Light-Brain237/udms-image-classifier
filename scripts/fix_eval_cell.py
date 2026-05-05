"""Fix cell 23: rebuild model + load_weights + confusion matrix."""
import json

NEW_SOURCE = """\
# ── Rebuild model + load Phase-2 weights ──────────────────────────────────
# PHASE2_CKPT is a WEIGHTS-ONLY file (.weights.h5).
# We must rebuild the exact same architecture, then call load_weights().
# Never use load_model() on a weights-only checkpoint.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = [
    'bad_drainage', 'damaged_signage', 'illegal_dumping',
    'potholes', 'vegetation_overgrowth',
]
NUM_CLASSES = 5
INPUT_SHAPE = (224, 224, 3)

def _rebuild_model() -> tf.keras.Model:
    \"\"\"Rebuild exact architecture used during training.
    Rescaling(1/127.5, -1) is identical to mobilenet_v2.preprocess_input
    and is natively serializable — no Lambda, no custom_objects needed.
    \"\"\"
    base = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights='imagenet'
    )
    base.trainable = True  # match Phase-2 state so weight tensor shapes align

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name='image_input')
    x = tf.keras.layers.Rescaling(
        scale=1.0/127.5, offset=-1.0, name='mobilenetv2_preprocess'
    )(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation='softmax', name='predictions'
    )(x)
    return tf.keras.Model(inputs, outputs, name='udms_mobilenetv2')


best_model = _rebuild_model()
best_model.load_weights(PHASE2_CKPT)   # load_weights(), NOT load_model()
print(f'Weights loaded from: {PHASE2_CKPT}')

# ── Evaluate on test set ───────────────────────────────────────────────────
y_true_list, y_prob_list = [], []
for images, labels in test_ds:
    probs = best_model.predict(images, verbose=0)
    y_prob_list.append(probs)
    y_true_list.append(np.argmax(labels.numpy(), axis=1))  # one-hot -> int

y_true = np.concatenate(y_true_list)
y_prob = np.concatenate(y_prob_list)
y_pred = np.argmax(y_prob, axis=1)

test_acc = accuracy_score(y_true, y_pred)
print(f'Test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)')
print(f'Test samples  : {len(y_true):,}')
print()
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Test Accuracy: {test_acc:.2%}')
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
cells[23]['source'] = NEW_SOURCE

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cell 23 updated successfully.')
