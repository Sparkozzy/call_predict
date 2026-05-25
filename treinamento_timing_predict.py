# %%
import pandas as pd
import numpy as np
import os
import sys
import io
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fix para terminal Windows
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuração do MLflow
mlflow.set_tracking_uri("https://mlflow.mindflow-ia.com")
mlflow.set_experiment("timing_predict_v2")

# %%
# ── [SEÇÃO 1] Carregamento e Preparação dos Dados ──────────────────────────────────────────

DATA_PATH = os.path.join("data", "timing_predict_train_set.csv")
print(f"Carregando dados de: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Extração de features temporais do timestamp BRT
df['created_at_brt'] = pd.to_datetime(df['created_at_brt'])
df['hora_do_dia'] = df['created_at_brt'].dt.hour
df['dia_da_semana'] = df['created_at_brt'].dt.dayofweek

# Definindo Features e Target
target = 'is_success'
# Removendo o timestamp bruto
features = [
    'agent_id', 'n_tentativas', 'horas_desde_primeiro_contato', 
    'horas_desde_ultimo_contato', 'densidade_tentativas', 'pressao_recente', 
    'ddd', 'ls_probabilidade', 'hora_do_dia', 'dia_da_semana',
    'tentativas_mesma_hora', 'tentativas_mesmo_dia', 
    'mesma_hora_que_ultimo_contato', 'last_outcome', 'primeira_reason'
]

X = df[features].copy()
y = df[target].copy()

# Tratamento de Categóricas para o XGBoost
cat_cols = ['agent_id', 'ddd', 'dia_da_semana', 'last_outcome', 'primeira_reason']
for col in cat_cols:
    X[col] = X[col].astype('category')

# Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset de Treino: {X_train.shape}")
print(f"Dataset de Teste: {X_test.shape}")
print(f"Taxa de Sucesso (Treino): {y_train.mean():.2%}")

# %%
# ── [SEÇÃO 2] Treinamento com XGBoost e MLflow Autolog ────────────────────────────────────

# Cálculo do scale_pos_weight para lidar com desbalanceamento (se necessário)
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

with mlflow.start_run(run_name="XGBoost_Timing_Base"):
    mlflow.xgboost.autolog()
    
    # Parâmetros base
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': pos_weight,
        'n_estimators': 500,
        'enable_categorical': True,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Fit com Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Predições
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Métricas Adicionais
    auc_score = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("test_auc", auc_score)
    
    print("\n--- RESULTADOS DO MODELO ---")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, importance_type='gain')
    plt.title("Feature Importance (Gain)")
    plt.savefig("artifacts/tp_feature_importance.png")
    mlflow.log_artifact("artifacts/tp_feature_importance.png")
    
    # Salvar modelo final em formato JSON para produção
    MODEL_OUT = os.path.join("models", "timing_predict_v2_alpha.json")
    model.save_model(MODEL_OUT)
    mlflow.log_artifact(MODEL_OUT)
    
    print(f"\nModelo salvo em: {MODEL_OUT}")

# %%
