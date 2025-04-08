import functions
import matplotlib
matplotlib.use("Qt5Agg")


# Load data
img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = functions.load_dataset(img_folder, mask_folder, manual_folder)


# # Find the best n_clusters
# auc_mean, best_n_clusters, best_auc_mean = functions.find_best_n_clusters(X_all, Y_all, 2, 11)
# # Visualize all images when n_clusters= 2-6
# functions.visualize_all(X_all, 6)
# # Visualize a single image when n_clusters= 6
# functions.visualize_single(X_all[2], 6)


# # Use the optimized k-means algorithm
binary_seg, auc, vessel_clusters = functions.direction_enhanced_kmeans(X_all[3], Y_all[3], 4, 2)
functions.optimized_visualize(X_all[3], Y_all[3], binary_seg, auc)


# Comparison
# classic_means, optimized_means = functions.compare_kmeans_algorithms(X_all, Y_all, 6, 2)
