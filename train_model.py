# === 1. REQUIRED LIBRARIES ===
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# === 2. LOAD & CLEAN DATA ===
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Drop columns not useful for modeling
    drop_cols = [
        'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'gender',
        'street', 'city', 'state', 'zip', 'lat', 'long', 'dob', 'trans_num',
        'unix_time', 'merch_lat', 'merch_long'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Store original category labels before encoding
    merchant_labels = df['merchant'].astype('category').cat.categories.tolist()
    category_labels = df['category'].astype('category').cat.categories.tolist()
    job_labels = df['job'].astype('category').cat.categories.tolist()

    # Encode categorical columns
    df['merchant'] = df['merchant'].astype('category').cat.codes
    df['category'] = df['category'].astype('category').cat.codes
    df['job'] = df['job'].astype('category').cat.codes

    # Scale amount
    scaler = StandardScaler()
    df['scaled_amt'] = scaler.fit_transform(df[['amt']])
    df.drop('amt', axis=1, inplace=True)

    return df, scaler, merchant_labels, category_labels, job_labels

# === 3. TRAIN MODELS & SAVE ===
def train_models(df, scaler, merchant_labels, category_labels, job_labels, model_dir='models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Save feature columns
    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, os.path.join(model_dir, "feature_columns.pkl"))

    # Save label lists
    joblib.dump(merchant_labels, os.path.join(model_dir, "merchant_labels.pkl"))
    joblib.dump(category_labels, os.path.join(model_dir, "category_labels.pkl"))
    joblib.dump(job_labels, os.path.join(model_dir, "job_labels.pkl"))

    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Define models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10, random_state=42)
    }

    scores = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_resampled, y_resampled)
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(f"✅ ROC-AUC Score for {name}: {roc_auc:.4f}")
        scores[name] = roc_auc

    return scores

# === 4. RUNNING SCRIPT ===
if __name__ == "__main__":
    dataset_path = "../data/fraud_dataset.csv"  # Adjust path if needed
    df, scaler, merchant_labels, category_labels, job_labels = load_and_prepare_data(dataset_path)
    model_scores = train_models(df, scaler, merchant_labels, category_labels, job_labels)

    best_model = max(model_scores, key=model_scores.get)
    print(f"\n🎯 Best performing model: {best_model.upper()} (ROC-AUC: {model_scores[best_model]:.4f})")
