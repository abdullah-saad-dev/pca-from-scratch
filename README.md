# PCA from Scratch — Power Iteration

> **Math 2 Project** · Principal Component Analysis implemented without any linear-algebra shortcuts, validated against scikit-learn.

---

## What this project does

This script walks through every step of PCA by hand:

1. **Standardise** — rescale every feature to mean 0, std 1
2. **Centre** — shift the data cloud to the origin
3. **Covariance matrix** — capture how every pair of features co-varies
4. **Power Iteration + Deflation** — find the top-*k* eigenvectors one at a time, without `np.linalg.eig`
5. **Project** — re-express the data along the new principal axes
6. **Validate** — compare results against `sklearn.decomposition.PCA`
7. **Visualise** — four plots saved as PNG files

**Dataset:** [Healthcare Disease Prediction Dataset](https://www.kaggle.com/datasets/algozee/healthcare-disease-prediction-dataset) (downloaded automatically via `kagglehub`).

---

## Repository structure

```
.
├── pca_power_iteration.py   # main script (all steps in one file)
├── README.md
├── slides/                  # PowerPoint slides explaining the algorithm
│   └── pca_explained.pptx
└── plots/                   # generated after running the script
    ├── plot_before_pca.png
    ├── plot_manual_pca.png
    ├── plot_sklearn_pca.png
    └── plot_variance_ratio.png
```

---

## Prerequisites

- Python **3.9 or later**
- A [Kaggle account](https://www.kaggle.com/) with an API token configured (needed by `kagglehub` to download the dataset)

### Configure Kaggle credentials

1. Go to [kaggle.com → Settings → API](https://www.kaggle.com/settings) and click **Create New Token**.
2. A file called `kaggle.json` will be downloaded. Place it at:
   - **Linux / macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<YourUser>\.kaggle\kaggle.json`
3. Restrict its permissions (Linux/macOS only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

If you prefer to install packages manually:

```bash
pip install numpy pandas matplotlib scikit-learn kagglehub
```

---

## Running the script

```bash
python pca_power_iteration.py
```

Expected console output (numbers will vary slightly by run):

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

Four PNG plots are saved in the working directory.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Matrix operations, random seed, norms |
| `pandas` | CSV loading, results table |
| `matplotlib` | Scatter plots and variance chart |
| `scikit-learn` | `PCA` class used for validation only |
| `kagglehub` | Automatic dataset download from Kaggle |

---

## How Power Iteration works (short version)

Given a covariance matrix **C**, we want its eigenvector with the largest eigenvalue.

1. Start with a random unit vector **v**.
2. Repeatedly compute **v ← C·v / ‖C·v‖**.
3. The direction converges to the dominant eigenvector because each multiplication amplifies the dominant direction more than any other.
4. The eigenvalue is recovered as **λ = vᵀ·C·v**.
5. **Deflation:** subtract **λ·vvᵀ** from **C** to erase that direction, then repeat for the next component.

For full mathematical details see the slides in `slides/`.

---

## Results

The manual implementation matches scikit-learn to floating-point precision (differences on the order of 10⁻¹²), confirming correctness of the Power Iteration + Deflation approach.

---

## License

This project was created for academic purposes. Feel free to use or adapt it.

