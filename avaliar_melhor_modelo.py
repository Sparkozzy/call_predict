import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
import sys
import io
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Fix para console no Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurar MLflow para apontar para o remoto
os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.mindflow-ia.com"
mlflow.set_tracking_uri("https://mlflow.mindflow-ia.com")

# O ID da melhor Run (1º Lugar da nossa análise anterior)
RUN_ID = "8aa792ed41464d56b6ff525ad4f33487"

print("1. Carregando os dados e preparando o ambiente...")
DATA_PATH = "data/LS_training_data.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

# 1. Unificar last_disconnection_reason e last_reason em 'last_outcome'
df['last_outcome'] = df['last_disconnection_reason'].fillna(df['last_reason'])

# 2. Definir Features (removendo as redundantes) e Target
X = df.drop(columns=['target', 'last_disconnection_reason', 'last_reason'])
y = df['target']

# 3. Definir colunas categóricas
cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']

for col in cat_cols:
    X[col] = X[col].astype('category')

# Split Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("2. Treinando o melhor modelo (Parâmetros: max_depth=3, scale_pos_weight=82, learning_rate=0.084)...")
import xgboost as xgb

best_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.08429,
    'scale_pos_weight': 82,
    'tree_method': 'hist',
    'enable_categorical': True,
    'random_state': 42
}

model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

os.makedirs("artifacts/models", exist_ok=True)

# ---------------------------------------------------------
# 1. Feature Importance (Nativo do XGBoost usando Feature Importances)
# ---------------------------------------------------------
print("3. Gerando Feature Importance...")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:15] # Pegando o top 15

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette="viridis")
plt.title("Feature Importance Nativo (XGBoost) - Top 15")
plt.xlabel("Importância Relativa")
plt.ylabel("Variável")
plt.tight_layout()
plt.savefig("artifacts/models/melhor_modelo_feature_importance.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. Permutation Importance
# ---------------------------------------------------------
print("4. Calculando Permutation Importance (isso pode levar alguns segundos)...")
# O permutation importance bagunça os dados reais para ver o impacto na precisão.
# Sendo dados categóricos nativos, o sklearn consegue calcular isso normalmente.
r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

perm_sorted_idx = r.importances_mean.argsort()[-15:] # Top 15

plt.figure(figsize=(10, 8))
plt.boxplot(
    r.importances[perm_sorted_idx].T,
    vert=False,
    labels=X.columns[perm_sorted_idx]
)
plt.title("Permutation Importance (Dados de Teste) - Top 15")
plt.xlabel("Queda na Performance ao Embaralhar a Variável")
plt.tight_layout()
plt.savefig("artifacts/models/melhor_modelo_permutation_importance.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Matriz de Confusão
# ---------------------------------------------------------
print("5. Gerando Matriz de Confusão...")
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0 (Não Converteu)", "1 (Converteu)"])

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
plt.title("Matriz de Confusão - Melhor Modelo (Teste Set)")
plt.tight_layout()
plt.savefig("artifacts/models/melhor_modelo_matriz_confusao.png", dpi=300)
plt.close()

print("\n🚀 Todos os gráficos foram gerados com sucesso na pasta 'artifacts/models/'!")
