import functions
import matplotlib
matplotlib.use("Qt5Agg")


# Load data
img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = functions.load_dataset(img_folder, mask_folder, manual_folder)

# KMeans Clustering
functions.kmeans_clustering(X_all[2], Y_all[2], 6, 6, 8.5)


