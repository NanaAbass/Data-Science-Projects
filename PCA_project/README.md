# Patient Risk Phenotyping Using PCA & K-Means Clustering

A machine learning project that applies **Principal Component Analysis (PCA)** for dimensionality reduction and **K-Means Clustering** to identify distinct patient risk groups from clinical data — enabling data-driven, personalized healthcare strategies.

---

## Project Overview

Clinical datasets are often high-dimensional, making it difficult to identify meaningful patterns. This project tackles that challenge by:

- Reducing 11 clinical features into principal components that capture the most variance
- Clustering patients into distinct risk phenotypes based on their clinical profiles
- Visualizing and interpreting each cluster to uncover unique health risk patterns

---

## 📂 Project Structure

```
├── PCA.ipynb            # Main analysis notebook
├── clinical_data.csv    # Input dataset (500 patients, 11 features)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 🗂️ Dataset

The dataset (`clinical_data.csv`) contains **500 patient records** with the following **11 clinical features**:

| Feature | Description |
|---|---|
| `Age` | Patient age |
| `BMI` | Body Mass Index |
| `HbA1c_Result` | Glycated haemoglobin (diabetes indicator) |
| `Avg_Glucose_mgdL` | Average blood glucose level |
| `Systolic_BP_Reading1` | First systolic blood pressure reading |
| `Systolic_BP_Reading2` | Second systolic blood pressure reading |
| `Diastolic_BP` | Diastolic blood pressure |
| `LDL_Cholesterol` | Low-density lipoprotein cholesterol |
| `HDL_Cholesterol` | High-density lipoprotein cholesterol |
| `Creatinine_mgdL` | Kidney function marker |
| `Physical_Activity_Score` | Physical activity level score |

> ⚠️ **Note:** `BMI` and `LDL_Cholesterol` contain missing values (25 each), handled via mean imputation during preprocessing.

---

## Methodology

### 1. Data Cleaning & Preprocessing
- Checked dataset shape and structure (500 rows × 11 columns)
- Identified and imputed missing values using mean imputation
- Standardized features using `StandardScaler` to normalize the data before PCA

### 2. Principal Component Analysis (PCA)
- Applied PCA to reduce the 11-dimensional feature space
- Evaluated explained variance per component to determine optimal number of components
- Retained principal components capturing significant cumulative variance

### 3. K-Means Clustering
- Used the **Elbow Method** to identify the optimal number of clusters
- Applied K-Means on the PCA-reduced data to segment patients into distinct risk groups
- Analyzed cluster centroids to interpret the characteristics of each phenotype

### 4. Visualization
- Scatter plot of patient clusters in PCA space (PC1 vs PC2)
- Cluster centroid profiles to highlight distinguishing clinical features per group

---

## Key Results

- Patients were successfully segmented into distinct clusters, each reflecting a different **clinical risk profile**
- PCA effectively reduced dimensionality while retaining significant variance in the data
- Cluster interpretation revealed unique combinations of metabolic, cardiovascular, and lifestyle features per group, potentially indicating different risk levels or disease progression patterns

---

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the notebook:
```bash
jupyter notebook PCA.ipynb
```

---

## 🛠️ Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

> See `requirements.txt` for exact versions.

---

## Use Cases

- **Clinical Decision Support** — Help clinicians identify high-risk patient subgroups
- **Personalized Medicine** — Tailor interventions based on patient phenotype
- **Healthcare Resource Allocation** — Prioritize care for clusters with the most critical profiles
- **Epidemiological Research** — Understand population-level risk distributions

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Nana Kwame Abaasah**  
[GitHub](https://github.com/NanaAbass) · [LinkedIn](www.linkedin.com/in/nana-kwame-abaasah)
