# **Multi-class Disease Classification Project**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project develops a machine learning system to classify diseases based on patient demographic and laboratory data. We explore various classification algorithms, address data imbalances, and interpret our model's decisions to ensure reliability and understanding.

## Features

-   **Comprehensive Data Analysis:** In-depth exploration of patient demographic and laboratory test results.
-   **Advanced Preprocessing:** Techniques like feature scaling and SMOTE (Synthetic Minority Over-sampling Technique) to handle data inconsistencies and class imbalances.
-   **Multiple Machine Learning Models:** Implementation and comparison of:
    -   Logistic Regression
    -   Random Forest
    -   XGBoost
    -   A simple Neural Network
-   **Hyperparameter Tuning:** Optimization of the best-performing models (XGBoost) to achieve peak accuracy.
-   **Thorough Model Evaluation:** Assessment using classification reports, confusion matrices, and ROC-AUC curves.
-   **Model Interpretability with SHAP:** Utilizes SHAP (SHapley Additive Explanations) to understand how each feature contributes to the model's predictions, making the "black box" of AI more transparent.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

-   Python 3.6+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NanaAbass/Data-Science-Projects/Multiclass_Disease_Prediction_Project.git
    cd data_analytics/project02
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imblearn xgboost tensorflow shap
    ```

### Project Structure

```
.
├── Disease Classification.ipynb   # Main Jupyter Notebook with the project code
└── laboratory__data.csv           # Dataset used for disease classification
```

## Usage

To explore the project:

1.  **Open the Jupyter Notebook:** Launch Jupyter Lab or Jupyter Notebook and navigate to `Disease Classification.ipynb`.
2.  **Run Cells Sequentially:** Execute the cells in the notebook from top to bottom. This will perform:
    -   Data loading and initial exploration.
    -   Data preprocessing.
    -   Model training and comparison.
    -   Hyperparameter tuning.
    -   Model evaluation and interpretation.

## Dataset

The project utilizes the `laboratory__data.csv` dataset, which contains a variety of patient demographic information and laboratory test results. The dataset includes features such as:

-   `Gender`
-   `Age`
-   `Hemoglobin`
-   `RBC` (Red Blood Cell count)
-   `WBC` (White Blood Cell count)
-   `AST` (Aspartate Aminotransferase)
-   `ALT` (Alanine Aminotransferase)
-   `Cholesterol`
-   `Spirometry`
-   `Creatinine`
-   `Glucose`
-   `Lipase`
-   `Troponin`
-   `Disease` (The target variable, indicating the classified disease)

## Key Insights & Conclusion

-   **Model Performance:** The XGBoost model demonstrated high accuracy and F1-scores, effectively classifying diseases from non-linear laboratory data.
-   **Critical Features:** Hemoglobin, Glucose, and Troponin were identified as top predictors, aligning with clinical understanding (e.g., Glucose for Diabetes, Troponin for Heart Attack).
-   **Data Quality:** SMOTE proved essential in balancing the dataset, significantly improving recall for minority disease classes.

## Limitations & Future Work

-   **Static Data:** The current model uses a single snapshot of lab values. Future work could incorporate longitudinal data to improve predictive power.
-   **Deep Learning Exploration:** While XGBoost performed well, exploring more complex deep learning architectures, especially Transformer-based models for tabular data, could be beneficial with larger datasets.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
