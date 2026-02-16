from pathlib import Path

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

train_dir = resolve_train_dir()

classes = sorted([
    d.name for d in train_dir.iterdir()
    if d.is_dir()
])

def is_image(fn: str) -> bool:
    fn = fn.lower()
    return fn.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

print("Train dir:", train_dir)
print("Classes:", classes)

for c in classes:
    folder = train_dir / c
    n = sum(is_image(f) for f in os.listdir(folder))
    print(f"{c:12s} {n}")
