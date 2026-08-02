import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

FEATURES_DIR = "../results/features"
MODEL_PATH = "../results/svm_model.joblib"

# Load features and labels
X_train = np.load(os.path.join(FEATURES_DIR, "train_X.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "train_y.npy"))
X_valid = np.load(os.path.join(FEATURES_DIR, "valid_X.npy"))
y_valid = np.load(os.path.join(FEATURES_DIR, "valid_y.npy"))

# Train SVM (handle class imbalance with class_weight="balanced")
print(f"Train: {X_train.shape}, {np.bincount(y_train)} (labels)")
print(f"Valid: {X_valid.shape}, {np.bincount(y_valid)} (labels)")

svm = SVC(kernel="linear", class_weight="balanced", probability=True)
svm.fit(X_train, y_train)

# Save model
joblib.dump(svm, MODEL_PATH)
print(f"SVM model saved to {MODEL_PATH}")

# Evaluate on validation set
y_pred = svm.predict(X_valid)
print("Classification Report:\n", classification_report(y_valid, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_valid, y_pred))
