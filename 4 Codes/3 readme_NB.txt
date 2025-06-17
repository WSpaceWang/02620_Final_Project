DRIVE Vessel Segmentation using Naive Bayes
===========================================

This project uses the DRIVE dataset to segment retinal blood vessels using a custom Gaussian Naive Bayes classifier.

Folder Structure:
-----------------
training/
├── images/        -> Original RGB fundus images (.tif)
├── mask/          -> Binary masks indicating valid field of view (.gif)
└── 1st_manual/    -> Ground truth vessel annotations (.gif)

Dependencies:
-------------
- numpy
- opencv-python
- imageio
- matplotlib
- scikit-learn

Install using:
pip install numpy opencv-python imageio matplotlib scikit-learn

How It Works:
-------------
1. Loads DRIVE images and applies CLAHE for contrast enhancement.
2. Extracts labeled pixels within the mask region.
3. Balances the dataset by sampling equal background and vessel pixels.
4. Trains a custom Naive Bayes classifier.
5. Evaluates with classification report, confusion matrix, and ROC AUC.
6. Visualizes predictions by overlaying vessels in red on original images.

How to Run:
-----------
1. Download and extract the DRIVE dataset into the 'training/' directory.
2. Run the script:

   python vessel_segmentation.py

Outputs:
--------
- Prints evaluation metrics.
- Displays visual overlay of predicted vessels.