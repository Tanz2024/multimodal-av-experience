import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

def resolve_train_dir() -> Path:
    candidates = [
        BASE_DIR / "datasets" / "data" / "train",
        BASE_DIR / "datasets" / "gestures-hand" / "data" / "train",
        BASE_DIR / "datasets" / "data" / "data" / "train",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Could not find train directory. Checked: "
        + ", ".join(str(p) for p in candidates)
    )

TRAIN_DIR = resolve_train_dir()
OUT_DIR = BASE_DIR / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_image(fn: str) -> bool:
    fn = fn.lower()
    return fn.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

classes = sorted([
    d.name for d in TRAIN_DIR.iterdir()
    if d.is_dir()
])
print("Classes:", classes)

IMG_SIZE = 64
X, y = [], []
skip_train = {"none"}  # keep runtime "no hand" separate

for cls in classes:
    folder = TRAIN_DIR / cls
    imgs = [f for f in os.listdir(folder) if is_image(f)]

    if cls in skip_train:
        print(f"Skip '{cls}' for training (we use no-hand => none).")
        continue

    for f in tqdm(imgs, desc=f"Extract {cls}"):
        path = folder / f
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        feat = resized.astype(np.float32).flatten() / 255.0

        X.append(feat)
        y.append(cls)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=object)

print("X:", X.shape, "y:", y.shape)

np.save(OUT_DIR / "X.npy", X)
np.save(OUT_DIR / "y.npy", y)

with open(OUT_DIR / "classes_present.txt", "w") as f:
    for c in sorted(set(y.tolist())):
        f.write(c + "\n")

print(f"Saved to {OUT_DIR}: X.npy, y.npy, classes_present.txt")
