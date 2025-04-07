import os
import glob
import tifffile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


# 0 Load data
# Load full dataset
def load_dataset(img_folder, mask_folder, manual_folder):
    # Get all file paths
    img_files = sorted(glob.glob(os.path.join(img_folder, "*.tif")))
    mask_files = sorted(glob.glob(os.path.join(mask_folder, "*.gif")))
    manual_files = sorted(glob.glob(os.path.join(manual_folder, "*.gif")))
    # Ensure the number of files matches
    assert len(img_files) == len(mask_files) == len(manual_files), "Number of files in folders doesn't match."
    # Prepare dataset
    x_all = []
    y_all = []
    for img_f, mask_f, manual_f in zip(img_files, mask_files, manual_files):
        x, y = load_sample(img_f, mask_f, manual_f)
        x_all.append(x)
        y_all.append(y)
    return np.array(x_all), np.array(y_all)
# Load single sample
def load_sample(img_path, mask_path, manual_path):
    # Load all components
    img = tifffile.imread(img_path)
    mask = np.array(Image.open(mask_path))
    manual = np.array(Image.open(manual_path))
    # Apply mask to image (optional)
    img_masked = img.copy()
    for i in range(3):
        img_masked[:, :, i] = img[:, :, i] * (mask > 0)  # Assuming binary mask
    # Prepare for training
    x = img_masked.astype(np.float32)
    y = manual.astype(np.int32)
    # Standardization
    # x = (x - x.mean()) / x.std()
    # y = (y - y.mean()) / y.std()
    # Normalization
    x = x / 255.0
    y = y / 255.0
    return x, y


# 1 KMeans Clustering
# Clustering function
def kmeans_clustering(img_x, img_y, n_clusters, width, height):
    # Reshape the image data to (584*565, 3) {All with shape (584, 565, 3)}
    X = img_x.reshape(-1, 3)
    # Create mask for non-background pixels (background pixels have RGB values of 0)
    # Normalization: 0; Standardization: -1.1750538
    main_mask = np.any(X != 0, axis=1)
    X_main = X[main_mask]  # Keep only non-background pixels
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels_main = kmeans.fit_predict(X_main)
    # Rebuild complete label array (including background)
    labels = np.zeros(X.shape[0], dtype=int) - 1  # -1 for background
    labels[main_mask] = labels_main
    # Reshape labels back to original image size
    labels = labels.reshape(img_x.shape[:2])
    # Evaluate clustering results
    best_cluster, best_auc = evaluate_clusters(labels, img_y, n_clusters)
    # Visualize clustering results
    visualize_clustering(img_x, labels, labels_main, kmeans, n_clusters, width, height)
    return best_cluster, best_auc

# Evaluate
def evaluate_clusters(labels, img_y, n_clusters):
    best_auc = 0
    best_cluster = -1
    for i in range(n_clusters):
        # True vessel mask: set vessel pixels to 1, background to 0
        true_vessel = (img_y > 0).astype(np.int32).reshape(-1)
        # Prediction: set pixels in current cluster to 1, others to 0
        pred_vessel = (labels == i).astype(np.int32).reshape(-1)
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(true_vessel, pred_vessel)
        acc = accuracy_score(true_vessel, pred_vessel)
        recall_val = recall_score(true_vessel, pred_vessel)
        precision_val = precision_score(true_vessel, pred_vessel)
        f1 = f1_score(true_vessel, pred_vessel)
        # print(f"Cluster {i}:")
        # print("Confusion Matrix:")
        # print(cm)
        # print("Accuracy:", acc)
        # print("Recall (Sensitivity):", recall_val)
        # print("Precision:", precision_val)
        # print("F1 Score:", f1)
        try:
            auc_val = roc_auc_score(true_vessel, pred_vessel)
            # print("ROC AUC:", auc_val)
        except Exception as e:
            print("Unable to calculate ROC AUC, continuous prediction scores required")
            auc_val = 0  # Set default value to avoid undefined error
        print("")
        if auc_val > best_auc:
            best_auc = auc_val
            best_cluster = i
    print(f"Best matching cluster: {best_cluster}")
    print(f"Best AUC value: {best_auc:.4f}")
    return best_cluster, best_auc

# Visualization
def visualize_clustering(img_x, labels, labels_main, kmeans, n_clusters, width, height):
    plt.figure(figsize=(width, height))
    # Original Image
    plt.subplot(221)
    plt.imshow(img_x)
    plt.title("Original Image")
    plt.axis("off")
    # Clustered Image (Center Color)
    clustered_img_x = np.zeros_like(img_x)
    for i in range(n_clusters):
        mask = (labels == i).reshape(-1)
        clustered_img_x.reshape(-1, 3)[mask] = kmeans.cluster_centers_[i]
    plt.subplot(222)
    plt.imshow(np.clip(clustered_img_x, 0, 1))  # Ensure values are in [0, 1] range
    plt.title("Clustered Image (Center Color)")
    plt.axis("off")
    # Clustered Image (Colormap)
    plt.subplot(212)
    colormap = plt.imshow(labels, cmap="viridis")  # Use a colormap to visualize clusters
    plt.title("Clustered Image (Colormap)")
    plt.colorbar(colormap)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    # Print cluster sizes
    # unique_labels, counts = np.unique(labels_main, return_counts=True)
    # for label, count in zip(unique_labels, counts):
    #     print(f"Cluster {label} contains {count} pixels")