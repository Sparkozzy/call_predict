# %% 
# ── [SEÇÃO 1] Carregamento e Inspeção Inicial ──────────────────────────────────
import pandas as pd
import os
import sys
import io
import mlflow

# Fix para encoding no Windows (emojis do MLflow)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurações Iniciais
DATA_PATH = "data/LS_training_data.csv"
EXPERIMENT_NAME = "lead_scoring_v2"

# Garantir que pastas de artefatos existem
os.makedirs("artifacts/eda", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)

# Configurar MLflow
mlflow.set_tracking_uri("https://mlflow.mindflow-ia.com")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Carregando dados de: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n--- Informações Gerais ---")
print(f"Shape: {df.shape}")
df.info()

print("\n--- Primeiras Linhas ---")
print(df.head())

print("\n--- Distribuição do Target (target) ---")
print(df['target'].value_counts(normalize=True) * 100)
print(df['target'].value_counts())

# %%
# ── [SEÇÃO 2] Estatística Descritiva e Visualização do Target ─────────────────
import matplotlib
matplotlib.use('Agg') # Backend não interativo
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Estatísticas Descritivas (Numéricas) ---")
print(df.describe())

print("\n--- Cardinalidade (Categóricas) ---")
# Filtrar apenas colunas do tipo object
categoricas = df.select_dtypes(include=['object']).columns
for col in categoricas:
    print(f"{col}: {df[col].nunique()} valores únicos")

# Gráfico da Distribuição do Target
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='target', hue='target', palette='viridis', legend=False)
plt.title('Distribuição do Target (Engagement)')
plt.xlabel('Target (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')

# Adicionar labels de porcentagem
total = len(df)
for p in plt.gca().patches:
    height = p.get_height()
    percentage = '{:.1f}%'.format(100 * height/total)
    x = p.get_x() + p.get_width() / 2
    y = height
    plt.annotate(percentage, (x, y), ha='center', va='bottom')

plot_path = "artifacts/eda/01_distribuicao_target.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"\nGráfico salvo em: {plot_path}")

# %% 
# ── [SEÇÃO 3] Pré-processamento de Dados ──────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("\n--- Pré-processamento ---")

# 1. Unificar last_disconnection_reason e last_reason em 'last_outcome'
# Priorizamos o motivo técnico, mas usamos o status de negócio como fallback
df['last_outcome'] = df['last_disconnection_reason'].fillna(df['last_reason'])

# 2. Definir Features (removendo as redundantes) e Target
X = df.drop(columns=['target', 'last_disconnection_reason', 'last_reason'])
y = df['target']

# 3. Definir colunas categóricas
cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']

for col in cat_cols:
    print(f"Convertendo {col} para categoria...")
    X[col] = X[col].astype('category')

# Split Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# %%
# ── [SEÇÃO 4] Algoritmo de Otimização Evolutiva (10 Épocas) ──────────────────
import xgboost as xgb
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
import pandas as pd
import numpy as np
import random

# Ativar autolog (silencioso)
mlflow.sklearn.autolog(log_models=True, silent=True)

# Parâmetros Iniciais (Base)
best_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'scale_pos_weight': 32,
    'tree_method': 'hist',
    'enable_categorical': True,
    'random_state': 42
}

def get_variation(params):
    """Cria uma alteração aleatória nos parâmetros atuais."""
    new_params = params.copy()
    new_params['max_depth'] = max(3, params['max_depth'] + random.randint(-1, 2))
    new_params['learning_rate'] = max(0.01, min(0.3, params['learning_rate'] * random.uniform(0.7, 1.3)))
    new_params['scale_pos_weight'] = max(10, params['scale_pos_weight'] + random.randint(-5, 10))
    return new_params

melhor_fn = float('inf')
historico_otimizacao = []

print(f"\n🚀 Iniciando Otimização Evolutiva (10 Épocas)... Target: Menos Falsos Negativos")

for epoca in range(1, 11):
    print(f"\n--- ÉPOCA {epoca}/10 ---")
    
    # Em cada época, testamos 3 variações do melhor atual
    for i in range(3):
        # A primeira tentativa da primeira época é o baseline
        if epoca == 1 and i == 0:
            params_atuaiss = best_params
            tag = "baseline"
        else:
            params_atuaiss = get_variation(best_params)
            tag = f"mutacao_e{epoca}_v{i}"
        
        run_name = f"opt_epoca_{epoca}_{i}"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            clf = xgb.XGBClassifier(**params_atuaiss)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            cm = confusion_matrix(y_test, y_pred)
            fn = int(cm[1, 0])
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            # Logar metadados da otimização
            mlflow.set_tag("epoca", epoca)
            mlflow.set_tag("tipo_run", tag)
            mlflow.log_params(params_atuaiss)
            mlflow.log_metric("false_negatives", fn)
            mlflow.log_metric("recall", recall)
            
            print(f"[{tag}] FN: {fn} | Recall: {recall:.4f} | AUC: {auc:.4f}")
            
            # Critério de Seleção: Menos Falsos Negativos
            if fn < melhor_fn:
                print(f"✨ Novo melhor encontrado! FN reduziu de {melhor_fn} para {fn}")
                melhor_fn = fn
                best_params = params_atuaiss.copy()
            
            historico_otimizacao.append({
                "Epoca": epoca,
                "Tag": tag,
                "FN": fn,
                "Recall": recall,
                "AUC": auc,
                **params_atuaiss
            })

# Resultados Finais
df_otim = pd.DataFrame(historico_otimizacao)
print("\n--- ✅ OTIMIZAÇÃO CONCLUÍDA ---")
print(f"Melhor FN atingido: {melhor_fn}")
print("\nTop 5 Modelos da Otimização:")
print(df_otim.sort_values(by="FN").head(5)[["Epoca", "Tag", "FN", "Recall", "scale_pos_weight", "max_depth"]])

# Salvar e Logar Resultados
res_path = "artifacts/eda/otimizacao_10_epocas.csv"
df_otim.to_csv(res_path, index=False)
mlflow.log_artifact(res_path)

# %%
# ── [SEÇÃO 5] Salvamento do Melhor Modelo ─────────────────────────────────────
import json

print(f"\n--- 💾 Salvando Melhor Modelo para Avaliação Externa ---")

# Criar pasta para o modelo se não existir
os.makedirs("models", exist_ok=True)

# Treinar o modelo final com os melhores parâmetros
final_clf = xgb.XGBClassifier(**best_params)
final_clf.fit(X_train, y_train)

# Salvar o modelo em formato JSON (nativo do XGBoost)
model_path = "models/best_lead_scoring_model.json"
final_clf.save_model(model_path)

# Salvar os parâmetros para referência
params_path = "models/best_params.json"
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"✅ Melhor modelo salvo em: {model_path}")
print(f"✅ Parâmetros salvos em: {params_path}")
