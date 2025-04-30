# Retinal Vessel Segmentation using K-means Clustering

This repository contains a Python implementation of several K-means clustering algorithms for retinal vessel segmentation in fundus images.

## Project Overview

The project focuses on segmenting blood vessels in retinal images using various K-means clustering approaches:

1. Classic K-means Clustering: Basic implementation with optimal cluster selection
2. Optimized K-means Clustering: Merges multiple vessel-indicative clusters for improved segmentation
3. Direction-enhanced K-means Clustering: Incorporates gradient information (direction and magnitude) as additional features

## Key Files

- Kmeans_main.py: Main script that executes the workflow
- Kmeans_functions.py: Contains all necessary functions for data loading, processing, clustering, and visualization

## Dataset Structure

The code is designed to work with a specific dataset structure:
datasets/
├── training/
│ ├── images/ # Original retinal images (.tif)
│ ├── mask/ # FOV masks (.gif)
│ └── 1st_manual/ # Ground truth vessel annotations (.gif)

## Features

- Automatic Cluster Selection: Identifies optimal number of clusters by analyzing AUC performance
- Visualization Tools: Includes multiple visualization functions for image clustering results
- Performance Metrics: Evaluates segmentation quality using accuracy, sensitivity, specificity, F1-score, and AUC
- Image Augmentation: Optional color enhancement for improved vessel contrast

## Methodology

1. The system loads and preprocesses retinal images and their corresponding ground truth
2. K-means clustering is applied to segment the image into different clusters
3. Advanced methods identify which clusters correspond to vessel structures
4. Results are evaluated against ground truth annotations and visualized

## Implementation Details

The core K-means algorithm is implemented manually (without using sklearn) with these steps:
- Random initialization of cluster centers
- Iterative assignment of pixels to nearest cluster
- Recalculation of cluster centers
- Convergence based on objective function stabilization

The optimized version further identifies vessel clusters based on AUC scores and merges the top-performing clusters.

The direction-enhanced version incorporates gradient information as additional features to improve vessel detection, particularly for capturing directional characteristics of vessel structures.