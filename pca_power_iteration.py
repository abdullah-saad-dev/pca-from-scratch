"""Math 2 Project — PCA from Scratch (Power Iteration)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA   # for validation in Step 6
import kagglehub
import os
import glob

# =============================================================================
# DATA LOADING
# =============================================================================

path = kagglehub.dataset_download("algozee/healthcare-disease-prediction-dataset")
print(f"Dataset downloaded to: {path}")

csv_files = glob.glob(os.path.join(path, '*.csv'))
if csv_files:
    data_file = csv_files[0]
    print(f"Loading data from: {data_file}")
    dataset = pd.read_csv(data_file)
else:
    raise FileNotFoundError("No CSV files found in the downloaded dataset directory.")

numerical_cols = dataset.select_dtypes(include=[np.number]).columns
X = dataset[numerical_cols].to_numpy()   # work in pure numpy from here on

# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X = data.data

print(f"Features used: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# =============================================================================
# STEP 1 — STANDARDIZE
# =============================================================================
# Every feature is rescaled to mean=0, std=1 so no single feature
# dominates just because of its scale.
# Standardization = (x - mean) / std, applied column by column.

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)


# =============================================================================
# STEP 2 — CENTER
# =============================================================================
# Shift the data cloud so its center of mass sits at the origin.
# PCA draws axes through the origin, so this alignment is required.
# After standardization the mean is already near zero, but we don't
# take that for granted.

mean = np.mean(X_scaled, axis=0)
X_centered = X_scaled - mean


# =============================================================================
# STEP 3 — COVARIANCE MATRIX
# =============================================================================
# Build the (features × features) table that captures how every pair
# of features varies together across all samples.

cov_matrix = np.cov(X_centered, rowvar=False)


# =============================================================================
# STEP 4 — POWER ITERATION
# =============================================================================
#
# Power Iteration finds ONE eigenvector at a time by exploiting a simple
# fact: if you keep multiplying any vector by a matrix, it will eventually
# rotate to align with the matrix's dominant direction (the eigenvector
# with the largest eigenvalue).
#
# After finding each eigenvector we use DEFLATION — we subtract that
# eigenvector's contribution from the matrix so the next call to power
# iteration finds the SECOND largest direction, and so on.

def power_iteration(matrix, num_iterations=2000, tol=1e-12):
    """
    Find the dominant (largest) eigenvector of a symmetric matrix.

    Args:
        matrix        : square symmetric matrix (our covariance matrix)
        num_iterations: safety cap on the number of multiply-normalise loops
        tol           : stop early when the vector stops changing this much

    Returns:
        eigenvalue  : the variance captured along this direction (a scalar)
        eigenvector : the direction itself (a unit vector)
    """
    n = matrix.shape[0]

    # --- Seed ---
    # Start from a random unit vector.  Think of it as pointing in a
    # completely arbitrary direction before we let the matrix "pull" it.
    np.random.seed(42)                          # reproducibility
    vector = np.random.rand(n)
    vector = vector / np.linalg.norm(vector)    # make it length 1

    for _ in range(num_iterations):

        # --- Multiply ---
        # The matrix stretches the vector; directions the matrix "cares
        # about" get amplified more than others.
        new_vector = matrix @ vector

        # --- Normalise ---
        # Bring the stretched vector back to length 1 so it doesn't
        # blow up to infinity over many iterations.
        new_vector = new_vector / np.linalg.norm(new_vector)

        # --- Convergence check ---
        # Once the direction barely changes between iterations, we've
        # landed on the eigenvector.
        if np.linalg.norm(new_vector - vector) < tol:
            break

        vector = new_vector

    # --- Eigenvalue ---
    # Now that we have the eigenvector v, the eigenvalue λ satisfies
    # A·v = λ·v, so λ = v^T · A · v.
    eigenvalue = float(vector @ matrix @ vector)

    return eigenvalue, vector


def get_top_k_eigenpairs(cov_matrix, k):
    """
    Find the top-k eigenpairs using power iteration + deflation.

    Deflation works like this: once we've found the dominant direction,
    we REMOVE its contribution from the matrix (matrix -= λ·v·v^T).
    The modified matrix no longer "sees" that direction, so the next
    call to power_iteration finds the second-largest direction instead.

    Args:
        cov_matrix : the full covariance matrix
        k          : how many principal components we want

    Returns:
        eigenvalues  : 1-D array, length k, descending order
        eigenvectors : 2-D array, shape (features, k)
    """
    matrix       = cov_matrix.copy()   # work on a copy; don't mutate the original
    eigenvalues  = []
    eigenvectors = []

    for i in range(k):
        lam, vec = power_iteration(matrix)

        eigenvalues.append(lam)
        eigenvectors.append(vec)

        # --- Deflation ---
        # np.outer(vec, vec) builds the rank-1 matrix  v · v^T.
        # Subtracting lam * (v · v^T) removes exactly the component we
        # just found, leaving the remaining variance for the next round.
        matrix = matrix - lam * np.outer(vec, vec)

        print(f"  PC{i+1} found — eigenvalue: {lam:.6f}")

    eigenvalues  = np.array(eigenvalues)
    eigenvectors = np.column_stack(eigenvectors)   # shape: (n_features, k)

    return eigenvalues, eigenvectors


# --- Run it ---
k = 2
print(f"\nRunning power iteration to find top {k} principal components...")
eigenvalues_manual, eigenvectors_manual = get_top_k_eigenpairs(cov_matrix, k)


# =============================================================================
# STEP 5 — PROJECT
# =============================================================================
# Multiply the centered data by our eigenvector matrix.
# Each row (sample) is re-expressed using the new axes instead of the
# original feature axes.  Result shape: (n_samples, 2)

X_pca_manual = X_centered @ eigenvectors_manual

# Explained variance ratio: how much of the total variance does each PC hold?
total_variance          = np.sum(np.diag(cov_matrix))
explained_variance_manual = eigenvalues_manual / total_variance

print(f"\nManual PCA — Explained Variance Ratio:")
for i, ev in enumerate(explained_variance_manual):
    print(f"  PC{i+1}: {ev:.4f}  ({ev*100:.2f}%)")


# =============================================================================
# STEP 6 — SKLEARN COMPARISON
# =============================================================================

pca = PCA(n_components=k)
X_pca_sklearn = pca.fit_transform(X_scaled)

print(f"\nSklearn PCA — Explained Variance Ratio:")
for i, ev in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ev:.4f}  ({ev*100:.2f}%)")


# =============================================================================
# STEP 7 — SIGN ALIGNMENT + NUMERICAL COMPARISON
# =============================================================================
# Eigenvectors have no "correct" sign — both +v and −v are valid.
# Power iteration may lock onto the opposite direction vs sklearn.
# We align the manual result to sklearn's sign so plots and numbers match.

sign_alignment  = np.sign(X_pca_sklearn[0]) / np.sign(X_pca_manual[0])
X_pca_manual    = X_pca_manual * sign_alignment        # fix manual for plots too

X_pca_sklearn_aligned = X_pca_sklearn                 # sklearn is our reference

var_diff        = np.abs(explained_variance_manual - pca.explained_variance_ratio_)
projection_diff = np.abs(X_pca_manual - X_pca_sklearn_aligned)   # now both aligned

print(f"\nMax difference in Explained Variance Ratio : {np.max(var_diff):.2e}")
print(f"Max difference in Projected Coordinates   : {np.max(projection_diff):.2e}")

comparison_df = pd.DataFrame({
    'Metric'             : ['PC1 Variance', 'PC2 Variance'],
    'Manual (Power Iter)': explained_variance_manual,
    'Sklearn'            : pca.explained_variance_ratio_,
    'Absolute Difference': var_diff
})
print("\nComparison Table:")
print(comparison_df.to_string(index=False))


# =============================================================================
# STEP 8 — VISUALISATIONS
# =============================================================================

# --- Before PCA ---
plt.figure(figsize=(6, 4))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.4, s=10)
plt.xlabel("Feature 1 (standardised)")
plt.ylabel("Feature 2 (standardised)")
plt.title("Original Data — Before PCA (first two features)")
plt.tight_layout()
plt.savefig("plot_before_pca.png", dpi=150)
plt.show()

# --- After manual PCA ---
plt.figure(figsize=(6, 4))
plt.scatter(X_pca_manual[:, 0], X_pca_manual[:, 1],
            alpha=0.4, s=10, color="green")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("After PCA — Manual (Power Iteration)")
plt.tight_layout()
plt.savefig("plot_manual_pca.png", dpi=150)
plt.show()

# --- After sklearn PCA ---
plt.figure(figsize=(6, 4))
plt.scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1],
            alpha=0.4, s=10, color="red")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("After PCA — Sklearn")
plt.tight_layout()
plt.savefig("plot_sklearn_pca.png", dpi=150)
plt.show()

# --- Variance ratio — manual ---
plt.figure(figsize=(6, 4))
plt.plot(range(1, k + 1), explained_variance_manual,
         marker='o', linewidth=2, color="green", label="Manual")
plt.plot(range(1, k + 1), pca.explained_variance_ratio_,
         marker='s', linewidth=2, color="red",   label="Sklearn", linestyle="--")
plt.xticks(range(1, k + 1), [f"PC{i}" for i in range(1, k + 1)])
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Variance Captured per Principal Component")
plt.legend()
plt.tight_layout()
plt.savefig("plot_variance_ratio.png", dpi=150)
plt.show()

print("\nAll plots saved as PNG files.")