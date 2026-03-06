import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os

def train_and_save_model():
    """
    This function trains the disease classification model and saves the necessary
    artifacts (model, scaler, encoder) to the 'disease_predictor/model' directory.
    """
    print("Starting model training process...")

    # Define paths
    data_path = r'C:\Users\nanaa\gemini_tutorials\data_analytics\project02\laboratory__data.csv'
    model_dir = 'disease_predictor/model'
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load and Preprocess Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Encode categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    le = LabelEncoder()
    df['Disease_encoded'] = le.fit_transform(df['Disease'])

    # Separate features and target
    X = df.drop(columns=['Disease', 'Disease_encoded'])
    y = df['Disease_encoded']

    # 2. Train-test split
    # We will use the full dataset for training the final production model, 
    # but we maintain the split logic from the notebook for consistency.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 4. Handle Class Imbalance with SMOTE
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # 5. Train Final XGBoost Model
    # Using the best performing model from the notebook's initial evaluation
    print("Training XGBoost model...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train_resampled, y_train_resampled)
    
    # 6. Save Artifacts
    print(f"Saving model artifacts to '{model_dir}'...")
    joblib.dump(xgb, os.path.join(model_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'encoder.joblib'))
    
    print("Training complete and artifacts saved successfully!")
    print(f"  - {os.path.join(model_dir, 'model.joblib')}")
    print(f"  - {os.path.join(model_dir, 'scaler.joblib')}")
    print(f"  - {os.path.join(model_dir, 'encoder.joblib')}")

if __name__ == '__main__':
    train_and_save_model()
