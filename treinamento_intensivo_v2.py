# %% 
# ── [SEÇÃO 1] Carregamento e Configuração do Sistema Intensivo ────────────────
import pandas as pd
import numpy as np
import os
import sys
import io
import mlflow
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurações Iniciais
DATA_PATH = "data/LS_training_data_v2.csv"
EXPERIMENT_NAME = "lead_scoring_v2_intensive"

os.makedirs("artifacts/eda", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Configurar MLflow
mlflow.set_tracking_uri("https://mlflow.mindflow-ia.com")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Carregando dados otimizados para treinamento intensivo: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Garantir tipos categóricos
cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

X = df.drop(columns=['target'])
y = df['target']

# Split: Treino, Validação (para o loop), Teste (para veredito final)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# %%
# ── [SEÇÃO 2] Algoritmo Evolutivo de Treinamento Intensivo ─────────────────────

# Ativar autolog (silencioso)
mlflow.xgboost.autolog(log_models=True, silent=True)

# Parâmetros Iniciais (Base agressiva para Recall)
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
base_scale = n_neg / n_pos if n_pos > 0 else 1.0

best_params = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'scale_pos_weight': base_scale * 1.5, # Começamos já punitivos
    'tree_method': 'hist',
    'enable_categorical': True,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'early_stopping_rounds': 20
}

def get_variation(params):
    """Gera variações aleatórias para busca no espaço de hiperparâmetros."""
    new_params = params.copy()
    new_params['max_depth'] = random.choice([3, 4, 5, 6, 7])
    new_params['learning_rate'] = random.choice([0.01, 0.03, 0.05, 0.07, 0.1])
    # Variar o peso entre 1.0x e 3.0x a base desbalanceada
    new_params['scale_pos_weight'] = base_scale * random.uniform(1.0, 3.0)
    new_params['subsample'] = random.uniform(0.7, 1.0)
    new_params['colsample_bytree'] = random.uniform(0.7, 1.0)
    return new_params

# Variáveis de controle do "Melhor de Todos"
melhor_fn = float('inf')
melhor_precision = 0.0
melhor_modelo_final = None
melhores_params_finais = None

historico_otimizacao = []

print(f"\n🚀 Iniciando Sistema Intensivo (10 Épocas x 3 variações)...")
print(f"Meta Primária: Menor Falso Negativo | Meta Secundária: Maior Precisão")

for epoca in range(1, 11):
    print(f"\n--- 📅 ÉPOCA {epoca}/10 ---")
    
    for i in range(3):
        # Primeira run da primeira época é o baseline punitivo
        if epoca == 1 and i == 0:
            params_atuais = best_params.copy()
            tag = "baseline_recall"
        else:
            params_atuais = get_variation(best_params)
            tag = f"mutacao_e{epoca}_v{i}"
            
        run_name = f"intensive_e{epoca}_{i}"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            clf = xgb.XGBClassifier(**params_atuais)
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Avaliação no Val Set para decisão de evolução
            y_pred_val = clf.predict(X_val)
            cm_val = confusion_matrix(y_val, y_pred_val)
            fn_val = int(cm_val[1, 0])
            prec_val = precision_score(y_val, y_pred_val, zero_division=0)
            
            # Logar métricas customizadas
            mlflow.log_metric("val_fn", fn_val)
            mlflow.log_metric("val_precision", prec_val)
            mlflow.set_tag("strategy", "recall_punitive")
            
            print(f"[{tag}] FN: {fn_val} | Prec: {prec_val:.4f}")
            
            # CRITÉRIO DE SELEÇÃO (Punitivo FN > Precisão)
            is_better = False
            if fn_val < melhor_fn:
                is_better = True
            elif fn_val == melhor_fn and prec_val > melhor_precision:
                is_better = True
                
            if is_better:
                print(f"   ✨ Novo campeão! (FN: {fn_val}, Prec: {prec_val:.4f})")
                melhor_fn = fn_val
                melhor_precision = prec_val
                best_params = params_atuais.copy() # Evolui a base para a próxima época
                melhor_modelo_final = clf
                melhores_params_finais = params_atuais.copy()

            historico_otimizacao.append({
                "Epoca": epoca,
                "Tag": tag,
                "FN": fn_val,
                "Precision": prec_val,
                **params_atuais
            })

# %%
# ── [SEÇÃO 3] Veredito Final e Salvamento ─────────────────────────────────────

print("\n--- ✅ TREINAMENTO INTENSIVO CONCLUÍDO ---")
print(f"Melhor FN atingido: {melhor_fn}")
print(f"Precisão do melhor modelo: {melhor_precision:.4f}")

# Avaliação Final no Test Set (dados nunca vistos)
y_pred_test = melhor_modelo_final.predict(X_test)
y_proba_test = melhor_modelo_final.predict_proba(X_test)[:, 1]
cm_test = confusion_matrix(y_test, y_pred_test)

print("\n📊 Desempenho Final no Test Set (Hold-out):")
print(f"Falsos Negativos: {cm_test[1, 0]}")
print(f"Falsos Positivos: {cm_test[0, 1]}")
print(f"Recall Final: {recall_score(y_test, y_pred_test):.4f}")
print(f"Precision Final: {precision_score(y_test, y_pred_test):.4f}")
print(f"AUC Final: {roc_auc_score(y_test, y_proba_test):.4f}")

# Salvar
model_path = "models/best_intensive_model_v2.json"
melhor_modelo_final.save_model(model_path)

import json
with open("models/best_intensive_params.json", "w") as f:
    json.dump(melhores_params_finais, f, indent=4)

print(f"\n✅ Modelo Campeão salvo em: {model_path}")
