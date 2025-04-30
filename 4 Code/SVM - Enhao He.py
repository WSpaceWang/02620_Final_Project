import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from functions import load_dataset
from tqdm import tqdm

# RBF Kernel Function
def rbf_kernel(X1, X2, gamma):
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist)

# SVM Model
class SVCManual:
    def __init__(self, C=1.0, gamma=0.05, max_iter=100):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter

    def fit(self, X, y):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        m, n = X.shape
        y = y.astype(float)
        y[y == 0] = -1

        # Compute class weights
        class_counts = np.bincount(((y + 1) // 2).astype(int))
        total = len(y)
        class_weights = {
            1.0: total / (2 * class_counts[1]),
            -1.0: total / (2 * class_counts[0])
        }
        sample_weights = np.array([self.C * class_weights[label] for label in y])

        # Build QP problem
        K = rbf_kernel(X, X, self.gamma)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(m))
        G = matrix(np.vstack((-np.eye(m), np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), sample_weights)))
        A = matrix(y.reshape(1, -1))
        b = matrix(np.zeros(1))

        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        self.b = np.mean([
            y_k - np.sum(self.alpha * self.sv_y * rbf_kernel(np.array([x_k]), self.sv_X, self.gamma))
            for x_k, y_k in zip(self.sv_X, self.sv_y)
        ])

        # Store raw scores for Platt scaling
        self.proba_model = LogisticRegression()
        train_scores = self.project(X)
        self.proba_model.fit(train_scores.reshape(-1, 1), ((y + 1) // 2))

    def project(self, X):
        K = rbf_kernel(X, self.sv_X, self.gamma)
        return np.dot(K, self.alpha * self.sv_y) + self.b

    def predict(self, X):
        return (self.project(X) > 0).astype(int)

    def predict_proba(self, X):
        scores = self.project(X).reshape(-1, 1)
        return self.proba_model.predict_proba(scores)

# Load dataset
img_folder = "datasets/training/images"
mask_folder = "datasets/training/mask"
manual_folder = "datasets/training/1st_manual"
X_all, Y_all = load_dataset(img_folder, mask_folder, manual_folder, augment=False)

X_pixels, y_labels = [], []
for img, label in tqdm(zip(X_all, Y_all), total=len(X_all)):
    img_flat = img.reshape(-1, 3)
    label_flat = (label.reshape(-1) > 0).astype(np.int32)
    mask_flat = np.any(img_flat != 0, axis=1)
    X_pixels.append(img_flat[mask_flat])
    y_labels.append(label_flat[mask_flat])

X = np.vstack(X_pixels)
y = np.concatenate(y_labels)

# Sample subset for speed
sample_size = 8000
idx = np.random.choice(len(X), size=sample_size, replace=False)
X, y = X[idx], y[idx]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train manual SVC
clf = SVCManual(C=1.0, gamma=0.05)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
