import numpy as np
import pandas as pd
import functions
import matplotlib
matplotlib.use("Qt5Agg")


# Load data
img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = functions.load_dataset(img_folder, mask_folder, manual_folder, augment=True)


# Find the best n_clusters
auc_mean, best_n_clusters, best_auc_mean = functions.find_best_n_clusters(X_all, Y_all, 2, 11)
# Visualize all images when n_clusters= 2-10
functions.visualize_all(X_all, 10)
# Visualize a single image when n_clusters= 6
functions.visualize_single(X_all[2], 10)


# Use the optimized k-means algorithm
results = []
for i in np.arange(len(X_all)):
    binary_seg, auc, vessel_clusters = functions.optimized_kmeans(X_all[i], Y_all[i], 10, 3)
    results.append((i+1, auc))
results = pd.DataFrame(sorted(results, key=lambda x: x[1], reverse=True), columns=['Index', 'AUC'])
# results.to_csv("results_test/optimized_auc.csv", index=False)
# Visualize the top 3 results
for i in np.arange(3):
    best_index = int(results.iloc[i]["Index"]) - 1
    binary_seg, auc, vessel_clusters = functions.optimized_kmeans(X_all[best_index], Y_all[best_index], 10, 3)
    functions.optimized_visualize(X_all[best_index], Y_all[best_index], binary_seg, auc)


# Comparison
classic_means, optimized_means = functions.compare_kmeans_algorithms(X_all, Y_all, 10, 3)


# Direction-enhanced KMeans Clustering
binary_seg, auc, vessel_clusters = functions.direction_enhanced_kmeans(X_all[2], Y_all[2], 10, 5)
functions.optimized_visualize(X_all[2], Y_all[2], binary_seg, auc)
