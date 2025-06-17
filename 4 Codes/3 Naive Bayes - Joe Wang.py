import os
import numpy as np
from glob import glob
import cv2
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ----- CLAHE for contrast enhancement -----
def apply_clahe_to_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# ----- Load DRIVE Data -----
def load_drive_data(img_folder, mask_folder, manual_folder, use_clahe=True):
    img_paths = sorted(glob(os.path.join(img_folder, "*.tif")))
    mask_paths = sorted(glob(os.path.join(mask_folder, "*.gif")))
    manual_paths = sorted(glob(os.path.join(manual_folder, "*.gif")))

    X_all = []
    Y_all = []

    for img_path, mask_path, label_path in zip(img_paths, mask_paths, manual_paths):
        print(f"Loading: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if use_clahe:
            img = apply_clahe_to_rgb(img)

        mask = imread(mask_path)
        label = imread(label_path)
        mask = (mask > 0).astype(np.uint8)
        label = (label > 0).astype(np.uint8)

        valid_pixels = mask > 0
        img_pixels = img[valid_pixels]
        label_pixels = label[valid_pixels]

        X_all.append(img_pixels)
        Y_all.append(label_pixels)

    X_all = np.vstack(X_all)
    Y_all = np.hstack(Y_all)
    return X_all, Y_all

# ----- Balance dataset -----
def balance_dataset(X, Y, ratio=1.0):
    vessel_idx = np.where(Y == 1)[0]
    background_idx = np.where(Y == 0)[0]
    np.random.shuffle(background_idx)
    background_idx = background_idx[:int(len(vessel_idx) * ratio)]
    selected_idx = np.concatenate([vessel_idx, background_idx])
    np.random.shuffle(selected_idx)
    return X[selected_idx], Y[selected_idx]

# ----- Naive Bayes Classifier -----
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_prob(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_sample(self, x):
        posteriors = {}
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._gaussian_prob(c, x)))
            posteriors[c] = prior + conditional
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

# ----- Visualization -----
def visualize_predictions(original_img, mask, y_pred, title="Prediction"):
    pred_img = np.zeros(mask.shape, dtype=np.uint8)
    pred_img[mask > 0] = y_pred

    overlay = original_img.copy()
    overlay[pred_img == 1] = [255, 0, 0]  # Red overlay

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[1].imshow(pred_img, cmap='gray')
    axs[1].set_title("Predicted Vessels")
    axs[2].imshow(overlay)
    axs[2].set_title("Overlay (Red = Vessel)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# ----- Main Execution -----
if __name__ == "__main__":
    img_folder = "training/images"
    mask_folder = "training/mask"
    manual_folder = "training/1st_manual"

    # Load and balance data
    X_all, Y_all = load_drive_data(img_folder, mask_folder, manual_folder)
    print("Original dataset shape:", X_all.shape, Y_all.shape)

    X_balanced, Y_balanced = balance_dataset(X_all, Y_all, ratio=1.0)
    print("Balanced dataset shape:", X_balanced.shape, Y_balanced.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, Y_balanced, test_size=0.2, random_state=42
    )

    # Train model
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    except ValueError:
        print("ROC AUC not defined (only one class predicted)")

    # ----- Visualize prediction on one image -----
    print("\nVisualizing prediction on a sample image...")

    sample_img_path = sorted(glob(os.path.join(img_folder, "*.tif")))[0]
    sample_mask_path = sorted(glob(os.path.join(mask_folder, "*.gif")))[0]

    sample_img = cv2.imread(sample_img_path)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img = apply_clahe_to_rgb(sample_img)

    sample_mask = imread(sample_mask_path)
    sample_mask = (sample_mask > 0).astype(np.uint8)
    sample_pixels = sample_img[sample_mask > 0]

    sample_preds = model.predict(sample_pixels)

    visualize_predictions(sample_img, sample_mask, sample_preds)
