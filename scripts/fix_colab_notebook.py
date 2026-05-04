"""Fix garbled cells in colab_train.ipynb."""
import json

with open('notebooks/colab_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Fix Cell 11: cache dir + corrupt scan ────────────────────────────────────
cells[11]['source'] = """\n# tf.data disk cache — images read from Drive once (epoch 1), cached to local SSD.
# From epoch 2 onward all reads come from fast local disk.
import os

CACHE_DIR = '/content/tf_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
print(f'tf.data cache dir : {CACHE_DIR}')
print('Epoch 1 warms cache from Drive; epoch 2+ reads from local SSD.')

# ── Scan and remove corrupt/unreadable files before training ─────────────────
# TensorFlow crashes on WebP, SVG, or truncated files — remove them first.
from PIL import Image as _PIL_Image
import pathlib as _pl

_bad_files = []
for _p in _pl.Path(DATA_DIR).rglob('*'):
    if not _p.is_file():
        continue
    try:
        with _PIL_Image.open(_p) as _im:
            _im.verify()
    except Exception:
        _bad_files.append(_p)

if _bad_files:
    print(f'Removing {len(_bad_files)} corrupt/unreadable files...')
    for _p in _bad_files:
        _p.unlink()
        print(f'  removed: {_p.name}')
else:
    print('No corrupt files found — dataset is clean.')
""".splitlines(keepends=True)

# ── Fix Cell 12: augmentation + make_dataset (flat 80/20 split) ──────────────
cells[12]['source'] = """\n# ── Augmentation layers (training only, on GPU inside tf.data) ──────────────
_augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name='augmentation')


def make_dataset(subset: str) -> tf.data.Dataset:
    \"\"\"Load training or validation split from flat DATA_DIR with 80/20 split.

    Args:
        subset: 'training' or 'validation'.

    Returns float32 images in [0, 255].
    Augmentation applied only to training subset.
    Preprocessing to [-1, 1] is handled by the model Lambda layer.
    \"\"\"
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VAL_SPLIT,
        subset=subset,
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode='categorical',
    )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Cache to local SSD — Drive only read on epoch 1.
    # Cache BEFORE augmentation so augmentation re-randomises every epoch.
    ds = ds.cache(os.path.join(CACHE_DIR, subset))
    if subset == 'training':
        ds = ds.shuffle(4096, seed=SEED)
        ds = ds.map(
            lambda x, y: (_augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset('training')
val_ds   = make_dataset('validation')

# CLASS_NAMES is alphabetical — matches the index order from image_dataset_from_directory.
CLASS_NAMES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
print(f'Class names (index 0-{NUM_CLASSES-1}): {CLASS_NAMES}')


def count_samples(ds: tf.data.Dataset) -> int:
    return sum(int(x.shape[0]) for x, _ in ds)


n_train = count_samples(train_ds)
n_val   = count_samples(val_ds)

print(f'Train : {n_train:>5,} images  (80% of dataset)')
print(f'Val   : {n_val:>5,} images  (20% of dataset)')
print(f'Batch shape: {next(iter(train_ds))[0].shape}')
""".splitlines(keepends=True)

# ── Fix Cell 23: evaluation — use val_ds (no test split) ─────────────────────
cells[23]['source'] = """\n# ── Load best checkpoint from Phase 2 ───────────────────────────────────────
best_model = tf.keras.models.load_model(PHASE2_CKPT)
print(f'Loaded: {PHASE2_CKPT}')

# ── Run inference on the validation set ──────────────────────────────────────
# No separate test split — the 20% validation set is used for final evaluation.
y_true_list, y_prob_list = [], []
for images, labels in val_ds:
    probs = best_model.predict(images, verbose=0)
    y_prob_list.append(probs)
    y_true_list.append(np.argmax(labels.numpy(), axis=1))

y_true = np.concatenate(y_true_list)
y_prob = np.concatenate(y_prob_list)
y_pred = np.argmax(y_prob, axis=1)

val_acc = accuracy_score(y_true, y_pred)
print(f'Validation accuracy : {val_acc:.4f}  ({val_acc * 100:.2f}%)')
print(f'Validation samples  : {len(y_true):,}')
print()
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
""".splitlines(keepends=True)

with open('notebooks/colab_train.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done. Cells 11, 12, 23 fixed.')
