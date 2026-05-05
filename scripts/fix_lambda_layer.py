"""Fix cells 17, 18, 20 — replace Lambda with Rescaling; fix cell 23 to use in-memory model."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── Cell 17: build_model — replace Lambda with Rescaling ─────────────────────
cells[17]['source'] = [
    "def build_model(freeze_backbone: bool = True) -> tf.keras.Model:\n",
    "    \"\"\"Build MobileNetV2 transfer-learning classifier.\n",
    "\n",
    "    Preprocessing (x/127.5 - 1) is applied via a Rescaling layer baked into\n",
    "    the model — fully serializable, no Lambda / custom_objects needed.\n",
    "    \"\"\"\n",
    "    base_model = tf.keras.applications.MobileNetV2(\n",
    "        input_shape=INPUT_SHAPE,\n",
    "        include_top=False,\n",
    "        weights='imagenet',\n",
    "    )\n",
    "    base_model.trainable = not freeze_backbone\n",
    "\n",
    "    inputs = tf.keras.Input(shape=INPUT_SHAPE, name='image_input')\n",
    "\n",
    "    # MobileNetV2 expects [-1, 1].  Rescaling(1/127.5, -1) is identical to\n",
    "    # mobilenet_v2.preprocess_input and is natively serializable by Keras 3.\n",
    "    x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0,\n",
    "                                   name='mobilenetv2_preprocess')(inputs)\n",
    "    x = base_model(x, training=False)\n",
    "    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)\n",
    "    x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)\n",
    "    x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)\n",
    "    x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)\n",
    "    outputs = tf.keras.layers.Dense(\n",
    "        NUM_CLASSES, activation='softmax', name='predictions'\n",
    "    )(x)\n",
    "\n",
    "    return tf.keras.Model(inputs, outputs, name='udms_mobilenetv2')\n",
    "\n",
    "\n",
    "model = build_model(freeze_backbone=True)\n",
    "model.summary(line_length=90)\n",
    "\n",
    "trainable     = sum(tf.size(w).numpy() for w in model.trainable_weights)\n",
    "non_trainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)\n",
    "print(f'\\nTrainable params     : {trainable:,}')\n",
    "print(f'Non-trainable params : {non_trainable:,}')\n",
]

# ── Cell 23: evaluation — use in-memory model (avoid load_model entirely) ────
cells[23]['source'] = [
    "# EarlyStopping(restore_best_weights=True) already restored the best weights\n",
    "# into `model` in memory — use it directly to avoid Lambda deserialization issues.\n",
    "best_model = model\n",
    "print('Using in-memory model with best weights restored by EarlyStopping.')\n",
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

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cells 17 and 23 fixed.')
