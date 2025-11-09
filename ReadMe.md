# 🏡 House Price Prediction Project  

### 📘 Overview
This project focuses on building and evaluating **regression models** to predict **house prices** using multiple property and neighborhood features. It serves as part of a data science portfolio demonstrating skills in **data exploration**, **machine learning**, and **model evaluation** using Python.  

---

## 🎯 Project Objectives
The main goals of this project are to:
1. Explore and understand the dataset.
2. Build and interpret a **Multiple Linear Regression** model.
3. Evaluate model performance on training and test sets.
4. Compare results with **Ridge (L2)** and **Lasso (L1)** regularized regression models.
5. Understand and handle issues such as **overfitting** and **multicollinearity**.

---

## 📂 Dataset
**File:** `house_prices_portfolio.csv`  
The dataset contains property-level information such as:
- Lot area and building size  
- Number of rooms and bathrooms  
- Neighborhood characteristics  
- Sale price (target variable)  

*(Note: The dataset may be simulated or anonymized for learning purposes.)*

---

## ⚙️ Technologies & Libraries Used
| Category | Tools/Libraries |
|-----------|----------------|
| **Programming Language** | Python 3.x |
| **Data Manipulation** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Modeling** | scikit-learn (`LinearRegression`, `Ridge`, `Lasso`) |
| **Evaluation Metrics** | Mean Squared Error (MSE), R² Score |

---

## 🧭 Project Workflow
1. **Data Loading & Cleaning**
   - Import dataset and inspect structure.
   - Handle missing values and encode categorical variables if necessary.

2. **Exploratory Data Analysis (EDA)**
   - Visualize relationships between variables.
   - Check distributions and correlations.
   - Identify multicollinearity using a correlation matrix.

3. **Model Development**
   - Split data into **training** and **testing** sets.
   - Train a **Multiple Linear Regression** model.
   - Evaluate performance using MSE and R².

4. **Regularization**
   - Apply **Ridge** and **Lasso** regressions to reduce overfitting.
   - Compare coefficients and predictive accuracy.

5. **Model Evaluation**
   - Analyze residual plots and diagnostic metrics.
   - Select the best-performing model based on generalization to test data.

---

## 📊 Key Learnings
- Importance of **feature selection** and **data preprocessing**.  
- How **regularization** improves model robustness.  
- Practical understanding of **bias-variance tradeoff**.  
- Use of Python’s data science ecosystem for end-to-end modeling.  

---

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/HousePrice_Project.git

2. Navigate to the project folder:

cd HousePrice_Project


3. Install dependencies:

pip install -r requirements.txt


4. Open the notebook:

jupyter notebook HousePrice_Project.ipynb

## 🧩 Folder Structure
HousePrice_Project/
│
├── HousePrice_Project.ipynb     # Main Jupyter notebook
├── house_prices_portfolio.csv   # Dataset file
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies

💡 Future Improvements

    - Open to suggestions and collaborations

## Author

Nana Abaasah
Data Scientist (in training) | Medical Laboratory Scientist | AWS Solutions Architect
📍 Ghana
🔗 LinkedIn

## License

This project is open-source and available under the MIT License.