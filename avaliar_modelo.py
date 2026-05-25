# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
import sys
import io

# %% [1] Imports e Configurações
# Fix para encoding no Windows (Emoji support no MLflow/XGBoost)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# Configurações de Caminhos
DATA_PATH = "data/LS_training_data.csv"
MODEL_PATH = "models/best_lead_scoring_model.json"
OUTPUT_DIR = "artifacts/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %% [2] Carregamento e Preparação de Dados
print("--- 🚀 Carregando Dados para Avaliação ---")
if not os.path.exists(DATA_PATH):
    print(f"❌ Erro: Arquivo de dados não encontrado em {DATA_PATH}.")
    sys.exit()

df = pd.read_csv(DATA_PATH, low_memory=False)

# Replicando Feature Engineering do treino
df['last_outcome'] = df['last_disconnection_reason'].fillna(df['last_reason'])
X = df.drop(columns=['target', 'last_disconnection_reason', 'last_reason'])
y = df['target']

# Tipagem Categórica
cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']
for col in cat_cols:
    X[col] = X[col].astype('category')

# Split consistente com o treino (usando random_state=42)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [3] Carregamento do Modelo
if not os.path.exists(MODEL_PATH):
    print(f"❌ Erro: Modelo não encontrado em {MODEL_PATH}. Execute o treinamento primeiro.")
    sys.exit()

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print(f"✅ Modelo carregado com sucesso de: {MODEL_PATH}")

# Gerar Probabilidades (base para todos os testes de threshold)
y_proba = model.predict_proba(X_test)[:, 1]

# %% [4] 🔥 TESTE DE THRESHOLD (ALTERE AQUI)
# Esta célula é dedicada para testes manuais. Altere o valor abaixo e execute.
THRESHOLD_TESTE = 0.15  # <--- Altere este valor livremente

y_pred_t = (y_proba >= THRESHOLD_TESTE).astype(int)
cm = confusion_matrix(y_test, y_pred_t)
recall = recall_score(y_test, y_pred_t)
precision = precision_score(y_test, y_pred_t, zero_division=0)
f1 = f1_score(y_test, y_pred_t, zero_division=0)

print(f"\n--- ⚖️ Monitoramento de Resultados (Threshold: {THRESHOLD_TESTE}) ---")
print(f"✅ Falsos Negativos (Perda de Oportunidade): {cm[1, 0]}")
print(f"⚠️ Falsos Positivos (Desperdício de Recurso): {cm[0, 1]}")
print(f"📊 Recall (Conversões Capturadas): {recall:.2%}")
print(f"🎯 Precision (Qualidade da Recomendação): {precision:.2%}")
print(f"📈 F1-Score: {f1:.2%}")

# Matriz de Confusão Visual
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Matriz de Confusão para Threshold {THRESHOLD_TESTE}')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# %% [5] Varredura Geral de Thresholds (Benchmarking)
print("\n--- 📊 Varredura de Performance (0.1 a 0.9) ---")
thresholds_range = np.arange(0.1, 1.0, 0.1)
results = []

for t in thresholds_range:
    y_p = (y_proba >= t).astype(int)
    results.append({
        "Threshold": round(t, 2),
        "FN": confusion_matrix(y_test, y_p)[1, 0],
        "FP": confusion_matrix(y_test, y_p)[0, 1],
        "Recall": recall_score(y_test, y_p),
        "Precision": precision_score(y_test, y_p, zero_division=0),
        "F1": f1_score(y_test, y_p, zero_division=0)
    })

df_metrics = pd.DataFrame(results)
print(df_metrics.to_string(index=False))

# Plot de Trade-off
plt.figure(figsize=(10, 6))
plt.plot(df_metrics['Threshold'], df_metrics['Recall'], marker='o', label='Recall (Sensibilidade)')
plt.plot(df_metrics['Threshold'], df_metrics['Precision'], marker='s', label='Precision (Confiança)')
plt.plot(df_metrics['Threshold'], df_metrics['F1'], marker='^', label='F1-Score (Equilíbrio)', linestyle='--')
plt.axvline(THRESHOLD_TESTE, color='red', linestyle=':', label=f'Threshold Atual ({THRESHOLD_TESTE})')
plt.title('Impacto do Threshold nas Métricas de Negócio')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [6] Distribuição Interativa de Probabilidades
print("\n--- 📈 Distribuição de Probabilidades ---")
plt.figure(figsize=(10, 6))
sns.kdeplot(y_proba[y_test == 0], label='Real: Não Converte (0)', fill=True, color='red', alpha=0.5)
sns.kdeplot(y_proba[y_test == 1], label='Real: Converte (1)', fill=True, color='green', alpha=0.5)
plt.axvline(THRESHOLD_TESTE, color='black', linestyle='--', label=f'Threshold Selecionado ({THRESHOLD_TESTE})')
plt.title('Distribuição de Probabilidades Previstas por Classe Real')
plt.xlabel('Score do Modelo (Probabilidade de Conversão)')
plt.ylabel('Densidade')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# %% [7] Análise de Leads Cortados vs Oportunidade Perdida
print(f"\n--- 📋 Análise de Volume (Threshold: {THRESHOLD_TESTE}) ---")

# Filtros baseados no threshold
leads_totais = len(y_test)
leads_cortados = np.sum(y_proba < THRESHOLD_TESTE)
falsos_negativos = np.sum((y_proba < THRESHOLD_TESTE) & (y_test == 1))
leads_recomendados = np.sum(y_proba >= THRESHOLD_TESTE)

print(f"Total de Leads em Teste: {leads_totais}")
print(f"Leads Cortados (Abaixo do Threshold): {leads_cortados} ({leads_cortados/leads_totais:.1%})")
print(f"Leads Recomendados (Acima do Threshold): {leads_recomendados} ({leads_recomendados/leads_totais:.1%})")
print(f"❌ Oportunidade Perdida (Leads que converteriam mas foram cortados): {falsos_negativos}")
taxa_erro = falsos_negativos/leads_cortados if leads_cortados > 0 else 0
print(f"⚠️ Taxa de Erro no Corte: {taxa_erro:.2%}")

# %% [8] Perfil dos Leads Cortados (O que estamos descartando?)
print(f"\n--- 🔍 Análise Qualitativa dos Leads Cortados (Threshold: {THRESHOLD_TESTE}) ---")

# Criando DataFrame temporário dos cortados para análise
idx_cortados = y_proba < THRESHOLD_TESTE
df_cortados = X_test[idx_cortados].copy()
df_cortados['target_real'] = y_test[idx_cortados]
df_cortados['score_modelo'] = y_proba[idx_cortados]

if len(df_cortados) > 0:
    print(f"\nTop 5 Motivos de Desconexão (last_outcome) nos Leads Cortados:")
    print(df_cortados['last_outcome'].value_counts().head(5))
    
    print(f"\nDistribuição de DDDs nos Leads Cortados:")
    print(df_cortados['ddd'].value_counts().head(5))
    
    print(f"\nMédia de Tentativas (tentativa_n) nos Leads Cortados: {df_cortados['tentativa_n'].mean():.2f}")
    
    print(f"\nAmostra de Leads que seriam DESCARTADOS (Cortados):")
    cols_view = ['tentativa_n', 'last_outcome', 'ddd', 'score_modelo', 'target_real']
    print(df_cortados[cols_view].head(10).to_string(index=False))
else:
    print("Nenhum lead cortado com o threshold atual.")

# %% [9] Exportação de Artefatos de Documentação
print("\n--- 🖼️ Salvando Gráficos para o Report ---")

# A. Distribuição de Scores (Salvar)
plt.figure(figsize=(10, 6))
sns.kdeplot(y_proba[y_test == 0], label='Classe 0 (Não Converte)', fill=True, color='red')
sns.kdeplot(y_proba[y_test == 1], label='Classe 1 (Converte)', fill=True, color='green')
plt.axvline(THRESHOLD_TESTE, color='black', linestyle='--', label=f'Corte Selecionado ({THRESHOLD_TESTE})')
plt.title('Distribuição de Probabilidades por Classe')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_distribuicao_prob.png"))
plt.close()

# B. Feature Importance
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[-15:]
plt.barh(range(len(indices)), importances[indices], color='skyblue')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Top 15 Feature Importances (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eval_feature_importance.png"))
plt.close()

print(f"✅ Artefatos finais salvos em: {OUTPUT_DIR}")

# %%
