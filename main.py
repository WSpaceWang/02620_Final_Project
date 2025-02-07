import os
import glob
import tifffile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def load_training_sample(img_path, mask_path, manual_path):
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


# Example usage for multiple files
def load_training_dataset(img_folder, mask_folder, manual_folder):
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
        x, y = load_training_sample(img_f, mask_f, manual_f)
        x_all.append(x)
        y_all.append(y)
    return np.array(x_all), np.array(y_all)


img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = load_training_dataset(img_folder, mask_folder, manual_folder)


"""----------------------------------------------------------------------"""


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

    # Calculate True Positive for each cluster (pixel-wise matching)
    # The ratio of the number of correctly identified blood vessel pixels to the number of true blood vessel pixels
    best_auc = 0
    best_cluster = -1
    for i in range(n_clusters):
        # 真实血管 mask：将血管像素设为 1，背景设为 0
        true_vessel = (img_y > 0).astype(np.int32).reshape(-1)
        # 预测结果：将 best_cluster 的像素设为 1，其余设为 0
        pred_vessel = (labels == i).astype(np.int32).reshape(-1)
        # 计算混淆矩阵
        cm = confusion_matrix(true_vessel, pred_vessel)
        acc = accuracy_score(true_vessel, pred_vessel)
        recall_val = recall_score(true_vessel, pred_vessel)
        precision_val = precision_score(true_vessel, pred_vessel)
        f1 = f1_score(true_vessel, pred_vessel)

        print(f"Cluster {i}:")
        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:", acc)
        print("Recall (Sensitivity):", recall_val)
        print("Precision:", precision_val)
        print("F1 Score:", f1)
        try:
            auc_val = roc_auc_score(true_vessel, pred_vessel)
            print("ROC AUC:", auc_val)
        except Exception as e:
            print("无法计算 ROC AUC，需要连续的预测得分")
        print("")
        if auc_val > best_auc:
            best_auc = auc_val
            best_cluster = i
    print(f"Best matching cluster: {best_cluster}")
    print(f"Best AUC value: {best_auc:.4f}")

# Back up
# # Calculate True Positive for each cluster (pixel-wise matching)
# # The ratio of the number of correctly identified blood vessel pixels to the number of true blood vessel pixels
# best_accuracy = 0
# best_cluster = -1
# accuracies = []
# vessel_pixels = img_y > 0  # Get ground truth vessel pixels
# total_vessel_pixels = np.sum(vessel_pixels)  # Calculate total vessel pixels
# for i in range(n_clusters):
#     mask_cluster = (labels == i).astype(np.int32)  # Create binary mask for current cluster
#     correct_predictions = np.sum((mask_cluster > 0) & vessel_pixels)  # Calculate correct predictions
#     accuracy = correct_predictions / total_vessel_pixels if total_vessel_pixels > 0 else 0
#     accuracies.append(accuracy)
#     # Update the best accuracy and cluster
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_cluster = i
# print(f"Full: {sum(accuracies)}")
# print(f"Best matching cluster: {best_cluster}")
# print(f"Best accuracy: {best_accuracy:.4f}")
# print("\nAccuracies for all clusters:")
# for i, acc in enumerate(accuracies):
#     print(f"Cluster {i}: {acc:.4f}")

    # Visualization
    plt.figure(figsize=(width, height))  # (12, 4) at first
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
    unique_labels, counts = np.unique(labels_main, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label} contains {count} pixels")


kmeans_clustering(X_all[2], Y_all[2], 6, 6, 8.5)


"""----------------------------------------------------------------------"""

# # Cluster Colors
#     plt.subplot(212)  # 使用整个底部行
#     for i in range(n_clusters):
#         plt.subplot(2, n_clusters, i + n_clusters + 1)  # 放在第二行
#         plt.imshow([[kmeans.cluster_centers_[i]]])
#         plt.title(f'Cluster {i}')
#         plt.axis('off')



















"""----------------------------------------------------------------------"""


















# # 假设X是形状为(20, 584, 565, 3)的数组
# # 需要先将图片数据重塑为二维数组，每张图片变成一个向量
# X_reshaped = X_all.reshape(20, -1)  # 将变成(20, 584*565*3)的形状
# # 2. 降采样（可选，为了提高速度）
# # X_reshaped = X_reshaped[:, ::10]  # 每10个像素点取1个
# # 使用K-means聚类，设置簇数为2
# kmeans = KMeans(n_clusters=2, random_state=0)
# kmeans.fit(X_reshaped)
# # 获取聚类标签
# labels = kmeans.labels_
# # 可视化聚类结果
# plt.figure(figsize=(10, 5))
# # 绘制聚类结果
# plt.subplot(1, 2, 1)
# plt.scatter(range(20), np.zeros(20), c=labels, cmap="viridis")
# plt.title("Image Clustering Results")
# plt.xlabel("Image Index")
# plt.yticks([])
# # 显示每个类别的图片数量
# plt.subplot(1, 2, 2)
# unique_labels, counts = np.unique(labels, return_counts=True)
# plt.bar(["Cluster "+str(i) for i in unique_labels], counts)
# plt.title("Number of Images in Each Cluster")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.show(block=True)
# # 打印每个类别的图片索引
# for i in range(2):
#     print(f"Cluster {i} contains images: {np.where(labels == i)[0]}")


"""----------------------------------------------------------------------"""







# # Display
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
# ax1.imshow(img)
# ax1.set_title("Original Image")
# ax2.imshow(mask)
# ax2.set_title("Mask")
# ax3.imshow(manual)
# ax3.set_title("Manual Reference")
# ax4.imshow(img_masked)
# ax4.set_title("Masked Image")
# plt.show(block=True)





