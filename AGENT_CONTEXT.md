# Agent Context — PCA from Scratch (Power Iteration)

## Goal
Execute the full PCA pipeline end-to-end: download data, standardise, compute the covariance matrix, find the top-2 principal components via Power Iteration + Deflation, project the data, validate against sklearn, and save four plots as PNG files.

---

## Environment

- Python 3.9+
- Required packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `kagglehub`
- Kaggle credentials must be present at `~/.kaggle/kaggle.json` before running

Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn kagglehub
```

---

## Script to run

```
pca_power_iteration.py
```

Run with:
```bash
python pca_power_iteration.py
```

---

## Workflow — step by step

### Step 0 — Download dataset
Use `kagglehub.dataset_download("algozee/healthcare-disease-prediction-dataset")`.
Find the first `.csv` file in the returned path and load it with `pandas`.
Extract all numerical columns into a numpy array `X`.

### Step 1 — Standardise
```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```
Every feature is rescaled to mean=0, std=1.

### Step 2 — Centre
```python
mean = np.mean(X_scaled, axis=0)
X_centered = X_scaled - mean
```
Shifts the data cloud so the origin is at the centre of mass.

### Step 3 — Covariance matrix
```python
cov_matrix = np.cov(X_centered, rowvar=False)
```
Shape: `(n_features, n_features)`.

### Step 4 — Power Iteration + Deflation
Implement two functions:

**`power_iteration(matrix, num_iterations=2000, tol=1e-12)`**
- Seed: `np.random.seed(42)`, random unit vector
- Loop: `new_vector = matrix @ vector`, normalise, check convergence (`norm(new_vector - vector) < tol`)
- Eigenvalue: `float(vector @ matrix @ vector)`
- Returns: `(eigenvalue, eigenvector)`

**`get_top_k_eigenpairs(cov_matrix, k)`**
- Loop `k` times: call `power_iteration`, append results, deflate: `matrix -= lam * np.outer(vec, vec)`
- Returns: `eigenvalues` (array, length k), `eigenvectors` (array, shape `(n_features, k)`)

Run with `k = 2`.

### Step 5 — Project
```python
X_pca_manual = X_centered @ eigenvectors_manual
```
Explained variance ratio:
```python
total_variance = np.sum(np.diag(cov_matrix))
explained_variance_manual = eigenvalues_manual / total_variance
```

### Step 6 — sklearn validation
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)
```

### Step 7 — Sign alignment + numerical comparison
Eigenvectors have no canonical sign. Align manual results to sklearn's sign:
```python
sign_alignment = np.sign(X_pca_sklearn[0]) / np.sign(X_pca_manual[0])
X_pca_manual   = X_pca_manual * sign_alignment
```
Compute and print:
- Max difference in explained variance ratio (expect ~1e-15)
- Max difference in projected coordinates (expect ~1e-12)
- A comparison DataFrame with columns: `Metric`, `Manual (Power Iter)`, `Sklearn`, `Absolute Difference`

### Step 8 — Save plots
Save four PNG files (150 dpi) using `plt.savefig(...)`:

| Filename | Content |
|---|---|
| `plot_before_pca.png` | Scatter of `X_scaled[:, 0]` vs `X_scaled[:, 1]` |
| `plot_manual_pca.png` | Scatter of manual PC1 vs PC2 (green) |
| `plot_sklearn_pca.png` | Scatter of sklearn PC1 vs PC2 (red) |
| `plot_variance_ratio.png` | Line plot comparing explained variance ratio, manual vs sklearn |

---

## Expected console output

```
Dataset downloaded to: ...
Loading data from: ...
Features used: <n>
Samples: <n>

Running power iteration to find top 2 principal components...
  PC1 found — eigenvalue: X.XXXXXX
  PC2 found — eigenvalue: X.XXXXXX

Manual PCA — Explained Variance Ratio:
  PC1: X.XXXX  (XX.XX%)
  PC2: X.XXXX  (XX.XX%)

Sklearn PCA — Explained Variance Ratio:
  PC1: X.XXXX  (XX.XX%)
  PC2: X.XXXX  (XX.XX%)

Max difference in Explained Variance Ratio : ~1e-15
Max difference in Projected Coordinates   : ~1e-12

All plots saved as PNG files.
```

---

## Success criteria

- [ ] Script runs without errors
- [ ] Max explained variance difference < 1e-10
- [ ] Max projection coordinate difference < 1e-8
- [ ] Four PNG files exist in the working directory

---

## Repository structure

```
pca-from-scratch/
├── README.md                    # human-facing setup and description
├── AGENT_CONTEXT.md             # this file — agent execution guide
├── pca_power_iteration.py       # main script
├── requirements.txt             # pip dependencies
├── slides/
│   └── pca_explained.pptx       # algorithm walkthrough deck
└── plots/                       # created after running the script
    ├── plot_before_pca.png
    ├── plot_manual_pca.png
    ├── plot_sklearn_pca.png
    └── plot_variance_ratio.png
```
