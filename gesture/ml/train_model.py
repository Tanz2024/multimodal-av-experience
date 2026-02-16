from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "artifacts"
X = np.load(OUT_DIR / "X.npy")
y = np.load(OUT_DIR / "y.npy", allow_pickle=True)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True, C=10, gamma="scale"))
])

model.fit(X_train, y_train)
pred = model.predict(X_val)

print(classification_report(y_val, pred))

joblib.dump(model, OUT_DIR / "gesture_model.joblib")
print(f"Saved model: {OUT_DIR / 'gesture_model.joblib'}")
