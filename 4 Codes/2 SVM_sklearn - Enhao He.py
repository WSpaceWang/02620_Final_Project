import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from functions import load_dataset
from tqdm import tqdm

# Load image, mask, and manual annotation
img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = load_dataset(img_folder, mask_folder, manual_folder, augment=False)

X_pixels = []
y_labels = []

# Extract RGB and labels from all pixels inside the image mask (ignore black background)
for img, label in tqdm(zip(X_all, Y_all), total=len(X_all)):
    h, w, _ = img.shape
    img_flat = img.reshape(-1, 3)
    label_flat = (label.reshape(-1) > 0).astype(np.uint8)  # 1: vessel, 0: background
    mask_flat = np.any(img_flat != 0, axis=1)  # exclude background

    X_pixels.append(img_flat[mask_flat])
    y_labels.append(label_flat[mask_flat])

# Merge all data into single array
X_pixels = np.vstack(X_pixels)
y_labels = np.concatenate(y_labels)

# Sample a subset for efficiency (optional, else memory explodes)
sample_size = 50000
idx = np.random.choice(len(X_pixels), size=sample_size, replace=False)
X_sample = X_pixels[idx]
y_sample = y_labels[idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)

# Train SVM
clf = SVC(kernel='rbf', probability=True, random_state=42,class_weight='balanced')
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred, target_names=["Background", "Vessel"]))
print("AUC:", roc_auc_score(y_test, y_proba))
