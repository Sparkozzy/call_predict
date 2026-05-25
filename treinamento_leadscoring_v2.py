# %% 
# ── [SEÇÃO 1] Carregamento e Configuração ─────────────────────────────────────
import pandas as pd
import os
import sys
import io
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_PATH = "data/LS_training_data_v2.csv"
EXPERIMENT_NAME = "lead_scoring_v2_xgboost_pro"

os.makedirs("artifacts/eda", exist_ok=True)
os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("https://mlflow.mindflow-ia.com")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Carregando dados otimizados de: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)

cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# %%
# ── [SEÇÃO 2] Preparação (Early Stopping Validation Set) ──────────────────────
# xgboost-lightgbm skill DO: Use early stopping with validation set.

X = df.drop(columns=['target'])
y = df['target']

# Divisão tripla: Train, Validation, Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Calcular pesos dinamicamente para lidar com classes desbalanceadas
# Aplicando um multiplicador de 2.0 para ser mais punitivo com Falsos Negativos (Prioridade: Recall)
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
scale_pos_weight = (n_neg / n_pos) * 2.0 if n_pos > 0 else 1.0

print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
print(f"Scale Pos Weight (Aggressive Recall): {scale_pos_weight:.2f}")

# %%
# ── [SEÇÃO 3] Treinamento XGBoost com Foco em Recall (Punitivo) ────────────────

mlflow.xgboost.autolog(log_models=True, silent=True)

# Hiperparâmetros baseados nas boas práticas
params = {
    'n_estimators': 500,        # Aumentado, pois usamos early stopping
    'max_depth': 5,             # Controlar overfitting
    'learning_rate': 0.05,      # Taxa menor para precisão
    'scale_pos_weight': scale_pos_weight,
    'tree_method': 'hist',
    'enable_categorical': True,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'early_stopping_rounds': 30
}

run_name = "xgb_recall_punitive_v2"
with mlflow.start_run(run_name=run_name):
    clf = xgb.XGBClassifier(**params)
    
    print("\nTreinando o modelo com FOCO EM RECALL (Punitivo)...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Previsão no TEST set (dados não vistos na validação)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Métricas
    cm = confusion_matrix(y_test, y_pred)
    fn = int(cm[1, 0])
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    mlflow.log_metric("false_negatives", fn)
    mlflow.log_metric("recall_test", recall)
    mlflow.log_metric("auc_test", auc)
    mlflow.log_metric("best_iteration", clf.best_iteration)
    
    print("\n--- Resultados no Conjunto de Teste ---")
    print(f"Falsos Negativos: {fn}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Melhor iteração (Early Stopping): {clf.best_iteration}")
    
    model_path = "models/best_lead_scoring_model_v2_pro.json"
    clf.save_model(model_path)
    print(f"✅ Modelo salvo em: {model_path}")

# %%
