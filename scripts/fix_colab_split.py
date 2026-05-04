"""Update colab_train.ipynb for 70/15/15 split."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Cell 9: Config — replace VAL_SPLIT with TEMP_SPLIT ───────────────────────
src9 = ''.join(cells[9]['source'])
src9 = src9.replace(
    "# 80/20 train/val split is done automatically in the data loading cell.",
    "# 70/15/15 train/val/test split is done automatically in the data loading cell."
)
src9 = src9.replace(
    "VAL_SPLIT       = 0.2\n",
    "TEMP_SPLIT      = 0.30   # val + test together = 30%;  train = 70%\n"
)
cells[9]['source'] = src9.splitlines(keepends=True)

# ── Cell 12: Data loading — 70/15/15 split ───────────────────────────────────
cells[12]['source'] = """\n# ── Augmentation layers (training only, on GPU inside tf.data) ──────────────
_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name='augmentation')


def _base_ds(subset: str) -> tf.data.Dataset:
    \"\"\"Load a subset (unbatched) from flat DATA_DIR using TEMP_SPLIT=0.30.\"\"\"
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=TEMP_SPLIT,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode='categorical',
    )
    return ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ── Training split (70%) ──────────────────────────────────────────────────────
train_raw = _base_ds('training')
train_raw = train_raw.cache(os.path.join(CACHE_DIR, 'train'))
train_raw = train_raw.shuffle(4096, seed=SEED)
train_raw = train_raw.map(
    lambda x, y: (_augment(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_raw.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── Temp split (30%) — split 50/50 into val (15%) and test (15%) ──────────────
temp_raw = _base_ds('validation')
temp_raw = temp_raw.cache(os.path.join(CACHE_DIR, 'temp'))
# Warm the cache before counting (required for tf.data cache to be seekable)
n_temp = sum(1 for _ in temp_raw)
n_val  = n_temp // 2

val_ds  = temp_raw.take(n_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = temp_raw.skip(n_val).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# CLASS_NAMES is alphabetical — matches index order from image_dataset_from_directory.
CLASS_NAMES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
print(f'Class names (index 0-{NUM_CLASSES-1}): {CLASS_NAMES}')


def count_samples(ds: tf.data.Dataset) -> int:
    return sum(int(x.shape[0]) for x, _ in ds)


n_train = count_samples(train_ds)
n_val_c = count_samples(val_ds)
n_test  = count_samples(test_ds)
total   = n_train + n_val_c + n_test

print(f'Train : {n_train:>5,} images  (~70%  → actual {n_train/total:.1%})')
print(f'Val   : {n_val_c:>5,} images  (~15%  → actual {n_val_c/total:.1%})')
print(f'Test  : {n_test:>5,} images  (~15%  → actual {n_test/total:.1%})')
print(f'Total : {total:>5,} images')
print(f'Batch shape: {next(iter(train_ds))[0].shape}')
""".splitlines(keepends=True)

# ── Cell 23: Evaluation — use test_ds for final report ───────────────────────
cells[23]['source'] = """\n# ── Load best checkpoint from Phase 2 ───────────────────────────────────────
best_model = tf.keras.models.load_model(PHASE2_CKPT)
print(f'Loaded: {PHASE2_CKPT}')

# ── Run inference on the held-out test set (15%) ─────────────────────────────
y_true_list, y_prob_list = [], []
for images, labels in test_ds:
    probs = best_model.predict(images, verbose=0)
    y_prob_list.append(probs)
    y_true_list.append(np.argmax(labels.numpy(), axis=1))

y_true = np.concatenate(y_true_list)
y_prob = np.concatenate(y_prob_list)
y_pred = np.argmax(y_prob, axis=1)

test_acc = accuracy_score(y_true, y_pred)
print(f'Test accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)')
print(f'Test samples  : {len(y_true):,}')
print()
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
""".splitlines(keepends=True)

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done. Cells 9, 12, 23 updated to 70/15/15 split.')
