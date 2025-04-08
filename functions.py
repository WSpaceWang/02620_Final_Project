import os
import glob
import tifffile
import numpy as np
import pandas as pd
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
    assert len(img_files) == len(mask_files) == len(manual_files), "Number of files in folders does not match."
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












# 2 Find the best n_clusters
def find_best_n_clusters(X_all, Y_all, min_clusters, max_clusters):
    auc_mean = pd.DataFrame(index=np.arange(min_clusters, max_clusters), columns=["n_clusters", "AUC_Mean"])
    for n in np.arange(min_clusters, max_clusters):
        auc = []
        for i in np.arange(len(X_all)):
            best_cluster, best_auc = kmeans(X_all[i], Y_all[i], n)
            auc.append(best_auc)
        auc_mean.loc[n, "n_clusters"] = n
        auc_mean.loc[n, "AUC_Mean"] = np.mean(auc)
    # Find the optimal number of clusters (with the highest mean AUC)
    best_idx = auc_mean["AUC_Mean"].idxmax()
    best_n_clusters = int(auc_mean.loc[best_idx, "n_clusters"])
    best_auc_mean = auc_mean.loc[best_idx, "AUC_Mean"]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(auc_mean.index, auc_mean["AUC_Mean"], marker=".", linestyle="-")
    plt.title("Mean AUC Values for Different Numbers of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Mean AUC")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(np.arange(min_clusters, max_clusters, 1))
    plt.tight_layout()
    plt.savefig("figures_test/1-1 Find the Best n_clusters.png", bbox_inches="tight")
    plt.show()
    return auc_mean, best_n_clusters, best_auc_mean


# 3 KMeans Clustering
# Clustering function
def kmeans(img_x, img_y, n_clusters):
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
        if auc_val > best_auc:
            best_auc = auc_val
            best_cluster = i
    return best_cluster, best_auc


# 4 Visualization
# Visualize single image
def visualize_single(img_x, n_clusters):
    X = img_x.reshape(-1, 3)
    main_mask = np.any(X != 0, axis=1)
    X_main = X[main_mask]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels_main = kmeans.fit_predict(X_main)
    labels = np.zeros(X.shape[0], dtype=int) - 1
    labels[main_mask] = labels_main
    labels = labels.reshape(img_x.shape[:2])
    # Plot
    plt.figure(figsize=(6, 8.5))
    # Original Image
    plt.subplot(221)
    plt.imshow(img_x)
    plt.title("Original Image")
    plt.axis("off")
    # Clustered Image (Center Color)
    clustered_img_x = np.zeros_like(img_x)
    for i in range(n_clusters):
        mask = (labels == i)
        for c in range(3):
            clustered_img_x[:, :, c][mask] = kmeans.cluster_centers_[i][c]
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
# Visualize all image
def visualize_all(X_all, n_clusters):
    fig, axes = plt.subplots(4, 5, figsize=(18, 16))
    axes = axes.flatten()
    for i in np.arange(len(X_all)):
        ax = axes[i]
        X = X_all[i].reshape(-1, 3)
        main_mask = np.any(X != 0, axis=1)
        X_main = X[main_mask]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels_main = kmeans.fit_predict(X_main)
        labels = np.zeros(X.shape[0], dtype=int) - 1
        labels[main_mask] = labels_main
        labels = labels.reshape(X_all[i].shape[:2])
        # Plot
        colormap = ax.imshow(labels, cmap="viridis")
        # ax.set_title(f"Image {i+1}")
        fig.colorbar(colormap, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
    fig.suptitle(f"Clustered Images (n_clusters={n_clusters})", fontsize=16)
    plt.tight_layout()
    plt.show()


# 5 Optimized KMeans Clustering
# Identify the most likely vessel clusters
def identify_vessel_clusters(labels, ground_truth, n_clusters, top_k):
    true_vessel = (ground_truth > 0).astype(np.int32).reshape(-1)
    scores = []
    for i in range(n_clusters):
        pred_vessel = (labels == i).astype(np.int32).reshape(-1)
        # Use the ROC AUC score to evaluate the cluster
        try:
            score = roc_auc_score(true_vessel, pred_vessel)
        except:
            score = 0
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    # Return the top_k clusters with the highest scores
    vessel_clusters = [cluster for cluster, _ in scores[:top_k]]
    return vessel_clusters, scores
# Merge the top clusters
def merge_top_clusters(labels, ground_truth, n_clusters, top_k):
    vessel_clusters, scores = identify_vessel_clusters(labels, ground_truth, n_clusters, top_k)
    # Create a mask for the identified vessel clusters
    vessel_mask = np.zeros_like(labels, dtype=bool)
    for cluster in vessel_clusters:
        vessel_mask |= (labels == cluster)
    # Create a segmentation mask
    segmentation = np.zeros_like(labels)
    segmentation[vessel_mask] = 1
    return segmentation, vessel_clusters, scores
# The optimized KMeans clustering function
def optimized_kmeans(img_x, img_y, n_clusters, top_k):
    X = img_x.reshape(-1, 3)
    main_mask = np.any(X != 0, axis=1)
    X_main = X[main_mask]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels_main = kmeans.fit_predict(X_main)
    labels = np.zeros(X.shape[0], dtype=int) - 1
    labels[main_mask] = labels_main
    labels = labels.reshape(img_x.shape[:2])
    # Merge the top clusters
    segmentation, vessel_clusters, scores = merge_top_clusters(labels, img_y, n_clusters, top_k)
    # Create a binary segmentation mask
    binary_segmentation = np.zeros_like(labels)
    for cluster in vessel_clusters:
        binary_segmentation[labels == cluster] = 1
    true_vessel = (img_y > 0).astype(np.int32).reshape(-1)
    pred_vessel = binary_segmentation.reshape(-1)
    try:
        auc_val = roc_auc_score(true_vessel, pred_vessel)
    except Exception as e:
        print("Unable to calculate ROC AUC, continuous prediction scores required")
        auc_val = 0
    return binary_segmentation, auc_val, vessel_clusters
# Separate visualization function
def optimized_visualize(img_x, img_y, binary_segmentation, auc_val, filename=None):
    plt.figure(figsize=(15, 5))
    # Original image
    plt.subplot(141)
    plt.imshow(img_x)
    plt.title("Original Image")
    plt.axis("off")
    # Ground truth vessel annotation
    plt.subplot(142)
    plt.imshow(img_y, cmap='gray')
    plt.title("Ground Truth Vessels")
    plt.axis("off")
    # Predicted vessel segmentation
    plt.subplot(143)
    plt.imshow(binary_segmentation, cmap='gray')
    plt.title(f"Predicted Vessels (AUC={auc_val:.4f})")
    plt.axis("off")
    # Overlay display (red for true positives, green for false positives, blue for false negatives)
    overlay = np.zeros((*binary_segmentation.shape, 3))
    # True Positives (TP): Red
    overlay[..., 0] = np.logical_and(binary_segmentation == 1, img_y > 0)
    # False Positives (FP): Green
    overlay[..., 1] = np.logical_and(binary_segmentation == 1, img_y == 0)
    # False Negatives (FN): Blue
    overlay[..., 2] = np.logical_and(binary_segmentation == 0, img_y > 0)
    plt.subplot(144)
    plt.imshow(overlay)
    plt.title("Overlay (Red=TP, Green=FP, Blue=FN)")
    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


# 6 Comparison
def compare_kmeans_algorithms(X_all, Y_all, n_clusters, top_k):
    classic_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'auc': []}
    optimized_metrics = {'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': [], 'auc': []}
    for i in range(len(X_all)):
        # Classic KMeans
        best_cluster, best_auc = kmeans(X_all[i], Y_all[i], n_clusters)
        # Get predictions from classic KMeans
        X = X_all[i].reshape(-1, 3)
        main_mask = np.any(X != 0, axis=1)
        X_main = X[main_mask]
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=0)
        labels_main = kmeans_model.fit_predict(X_main)
        labels = np.zeros(X.shape[0], dtype=int) - 1
        labels[main_mask] = labels_main
        labels = labels.reshape(X_all[i].shape[:2])
        classic_pred = (labels == best_cluster).astype(np.int32).reshape(-1)
        # Optimized KMeans
        binary_segmentation, opt_auc, _ = optimized_kmeans(X_all[i], Y_all[i], n_clusters, top_k)
        optimized_pred = binary_segmentation.reshape(-1)
        # True values
        true_vessel = (Y_all[i] > 0).astype(np.int32).reshape(-1)
        # Calculate metrics for both algorithms
        # Classic KMeans
        classic_metrics['accuracy'].append(accuracy_score(true_vessel, classic_pred))
        classic_metrics['sensitivity'].append(recall_score(true_vessel, classic_pred))
        classic_metrics['specificity'].append(calculate_specificity(true_vessel, classic_pred))
        classic_metrics['f1'].append(f1_score(true_vessel, classic_pred))
        classic_metrics['auc'].append(best_auc)
        # Optimized KMeans
        optimized_metrics['accuracy'].append(accuracy_score(true_vessel, optimized_pred))
        optimized_metrics['sensitivity'].append(recall_score(true_vessel, optimized_pred))
        optimized_metrics['specificity'].append(calculate_specificity(true_vessel, optimized_pred))
        optimized_metrics['f1'].append(f1_score(true_vessel, optimized_pred))
        optimized_metrics['auc'].append(opt_auc)
    # Calculate mean values
    classic_means = {k: np.mean(v) for k, v in classic_metrics.items()}
    optimized_means = {k: np.mean(v) for k, v in optimized_metrics.items()}
    # Create and display comparison table
    print("Performance Comparison: Classic KMeans vs. Optimized KMeans")
    print("-" * 60)
    print(f"{'Metric':<15} | {'Classic KMeans':<15} | {'Optimized KMeans':<15}")
    print("-" * 60)
    for metric in ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']:
        print(f"{metric.capitalize():<15} | {classic_means[metric]:.4f}{' ' * 10} | {optimized_means[metric]:.4f}")
    return classic_means, optimized_means
# Calculate specificity
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp)
    return specificity









































def direction_enhanced_kmeans(img_x, img_y, n_clusters, top_k):
    # 计算梯度信息来获取方向性特征
    from scipy import ndimage

    # 计算x和y方向的梯度
    grad_x = ndimage.sobel(img_x[:, :, 0], axis=1) + ndimage.sobel(img_x[:, :, 1], axis=1) + ndimage.sobel(
        img_x[:, :, 2], axis=1)
    grad_y = ndimage.sobel(img_x[:, :, 0], axis=0) + ndimage.sobel(img_x[:, :, 1], axis=0) + ndimage.sobel(
        img_x[:, :, 2], axis=0)

    # 计算梯度方向和强度
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_direction = np.arctan2(grad_y, grad_x)

    # 将原始RGB特征与方向和强度特征结合
    X_rgb = img_x.reshape(-1, 3)

    # 为每个像素添加方向和强度特征
    grad_magnitude_flat = grad_magnitude.reshape(-1, 1)
    grad_direction_flat = grad_direction.reshape(-1, 1)

    # 标准化新特征，避免它们主导聚类过程
    grad_magnitude_flat = (grad_magnitude_flat - np.min(grad_magnitude_flat)) / (
                np.max(grad_magnitude_flat) - np.min(grad_magnitude_flat) + 1e-10)

    # 将新特征与RGB特征结合
    X_enhanced = np.hstack(
        (X_rgb, grad_magnitude_flat * 0.5, np.sin(grad_direction_flat) * 0.3, np.cos(grad_direction_flat) * 0.3))

    # 创建非背景像素的掩码
    main_mask = np.any(X_rgb != 0, axis=1)
    X_main = X_enhanced[main_mask]

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels_main = kmeans.fit_predict(X_main)

    # 重新构建标签数组
    labels = np.zeros(X_rgb.shape[0], dtype=int) - 1
    labels[main_mask] = labels_main
    labels = labels.reshape(img_x.shape[:2])


    # 合并顶部聚类
    segmentation, vessel_clusters, scores = merge_top_clusters(labels, img_y, n_clusters, top_k)

    true_vessel = (img_y > 0).astype(np.int32).reshape(-1)
    pred_vessel = segmentation.reshape(-1)
    try:
        auc_val = roc_auc_score(true_vessel, pred_vessel)
    except Exception as e:
        print("Unable to calculate ROC AUC, continuous prediction scores required")
        auc_val = 0

    return segmentation, auc_val, vessel_clusters