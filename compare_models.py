import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

def load_data(path, is_v2=False):
    df = pd.read_csv(path, low_memory=False)
    if not is_v2:
        df['last_outcome'] = df['last_disconnection_reason'].fillna(df['last_reason'])
        X = df.drop(columns=['target', 'last_disconnection_reason', 'last_reason'])
    else:
        X = df.drop(columns=['target'])
    y = df['target']
    cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

_, X_test_v1, _, y_test_v1 = load_data("data/LS_training_data.csv", is_v2=False)
_, X_test_v2, _, y_test_v2 = load_data("data/LS_training_data_v2.csv", is_v2=True)

model_v1 = xgb.XGBClassifier()
if os.path.exists("models/best_lead_scoring_model.json"):
    model_v1.load_model("models/best_lead_scoring_model.json")
else:
    print("V1 model not found")

model_v2 = xgb.XGBClassifier()
if os.path.exists("models/best_intensive_model_v2.json"):
    model_v2.load_model("models/best_intensive_model_v2.json")
else:
    print("V2 intensive model not found")

def evaluate(model, X_test, y_test, threshold=0.15):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "FN": int(cm[1, 0]),
        "FP": int(cm[0, 1]),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba)
    }

print("=== V1 Metrics (Threshold 0.15) ===")
print(evaluate(model_v1, X_test_v1, y_test_v1))
print("=== V2 Metrics (Threshold 0.15) ===")
print(evaluate(model_v2, X_test_v2, y_test_v2))

print("\n=== Feature Importances V1 ===")
imp_v1 = pd.Series(model_v1.feature_importances_, index=X_test_v1.columns).sort_values(ascending=False)
print(imp_v1.head(10).to_string())

print("\n=== Feature Importances V2 ===")
imp_v2 = pd.Series(model_v2.feature_importances_, index=X_test_v2.columns).sort_values(ascending=False)
print(imp_v2.head(10).to_string())
