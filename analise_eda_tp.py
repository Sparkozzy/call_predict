# %%
import pandas as pd
import numpy as np
import os
import sys
import io
import xgboost as xgb
import matplotlib
matplotlib.use('Agg') # Backend não interativo
import matplotlib.pyplot as plt
import seaborn as sns

# Fix para console/emojis no Windows
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.makedirs("artifacts/eda", exist_ok=True)

# %%

DATA_PATH = os.path.join("data", "Retell_calls_V2_Final.csv")
MODEL_PATH = os.path.join("models", "best_intensive_model_v2.json")

print("1. Carregando dados base...")
df = pd.read_csv(DATA_PATH, low_memory=False, dtype={'to_number': str, 'from_number': str, 'call_id': str})

print("2. Aplicando Engenharia de Features...")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values(['to_number', 'created_at'])

# Deduplicar
df = df.dropna(subset=['disconnection_reason'])
df = df.drop_duplicates(subset=['call_id'], keep='last')

# Alinhamento temporal de fuso horário brasileiro para todo o dataset
df['created_at_brt'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert('America/Sao_Paulo')
df['hora_do_dia'] = df['created_at_brt'].dt.hour
df['dia_da_semana'] = df['created_at_brt'].dt.dayofweek


# Features Sequenciais (Calculadas antes dos filtros para manter o histórico completo)
df['n_tentativas'] = df.groupby('to_number').cumcount() # 0 para a primeira, 1 para a segunda...
df['tentativa_n'] = df['n_tentativas'] + 1 # Compatibilidade com modelo LS V2

# Tempo desde o início e desde a última chamada
primeira_chamada_time = df.groupby('to_number')['created_at'].transform('first')
df['horas_desde_primeiro_contato'] = (df['created_at'] - primeira_chamada_time).dt.total_seconds() / 3600.0

df['horas_desde_ultimo_contato'] = (df.groupby('to_number')['created_at'].diff().dt.total_seconds() / 3600.0).fillna(0)
df['hora_ultimo_contato'] = df.groupby('to_number')['created_at'].shift(1).dt.hour.fillna(-1)

# --- NOVAS FEATURES DE COMPORTAMENTO INDIVIDUAL (ANTI-VÍCIO HORÁRIO) ---
# 1. Quantidade de tentativas anteriores na mesma hora do dia
df['tentativas_mesma_hora'] = df.groupby(['to_number', 'hora_do_dia']).cumcount()

# 2. Quantidade de tentativas anteriores no mesmo dia da semana
df['tentativas_mesmo_dia'] = df.groupby(['to_number', 'dia_da_semana']).cumcount()

# 3. Indicador se a hora atual é idêntica à hora do último contato
df['mesma_hora_que_ultimo_contato'] = (df['hora_do_dia'] == df['hora_ultimo_contato']).astype(int)

# Features de Fadiga e Pressão (Conforme V1)
# densidade_tentativas: Média de tentativas por hora de vida do lead
df['densidade_tentativas'] = df['n_tentativas'] / (df['horas_desde_primeiro_contato'] + 1.0)

# pressao_recente: Relação entre volume total e proximidade da última chamada
df['pressao_recente'] = df['n_tentativas'] / (df['horas_desde_ultimo_contato'] + 1.0)

# Categorias e Outros
df['last_reason'] = df.groupby('to_number')['disconnection_reason'].shift(1).fillna('first_call')
primeira_reason_real = df.groupby('to_number')['disconnection_reason'].transform('first')
df['primeira_reason'] = np.where(df['tentativa_n'] == 1, 'first_call', primeira_reason_real)
df['ddd'] = df['to_number'].str.replace(r'\.0$', '', regex=True).str.extract(r'^55(\d{2})')
df['ddd'] = df['ddd'].fillna('desconhecido')
df['last_outcome'] = df['last_reason']

print(f"3. Aplicando inferência do modelo Lead Scoring V2...")
model_ls = xgb.Booster()
model_ls.load_model(MODEL_PATH)

features_ls = ['agent_id', 'tentativa_n', 'primeira_reason', 'horas_desde_primeiro_contato', 'ddd', 'last_outcome']
df_infer = df.copy()
cat_cols = ['agent_id', 'ddd', 'primeira_reason', 'last_outcome']
for col in cat_cols:
    df_infer[col] = df_infer[col].astype('category')

dtest = xgb.DMatrix(df_infer[features_ls], enable_categorical=True)
df['ls_probabilidade'] = model_ls.predict(dtest)

THRESHOLD = 0.4
df['is_high_score'] = (df['ls_probabilidade'] >= THRESHOLD)

# ── FILTRO DE LEADS REAIS E HIGH SCORE ─────────────────────────────────────────
real_leads = df[df['duration_s'] > 45]['to_number'].unique()
df_real = df[df['to_number'].isin(real_leads)].copy()
df_tp = df_real[df_real['is_high_score']].copy()

# Definindo o target de sucesso do Timing Predict
df_tp['is_success'] = ((df_tp['Marcada'] == 1) | (df_tp['duration_s'] > 45)).astype(int)

dias_label = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
print(f"Volume final para análise: {df_tp.shape[0]} registros")

# %%
# ── [SEÇÃO 2] Volume de Chamadas por Hora do Dia ──────────────────────────────────────────

plt.figure(figsize=(12, 5))
sns.countplot(data=df_tp, x='hora_do_dia', color='skyblue')
plt.title("Volume de Chamadas por Hora do Dia (Horário de Brasília - BRT)")
plt.xlabel("Hora do Dia (São Paulo)")
plt.ylabel("Quantidade de Chamadas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("artifacts/eda/tp_01_volume_hora.png")
plt.close()

# %%
# ── [SEÇÃO 3] Volume de Chamadas por Dia da Semana ────────────────────────────────────────

plt.figure(figsize=(10, 5))
sns.countplot(data=df_tp, x='dia_da_semana', color='salmon')
plt.title("Volume de Chamadas por Dia da Semana")
plt.xticks(ticks=range(7), labels=dias_label)
plt.xlabel("Dia da Semana")
plt.ylabel("Quantidade de Chamadas")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("artifacts/eda/tp_02_volume_dia.png")
plt.close()

# %%
# ── [SEÇÃO 5] Taxa de Sucesso por Hora do Dia ─────────────────────────────────────────────

success_rate = df_tp['is_success'].mean()
success_by_hour = df_tp.groupby('hora_do_dia')['is_success'].mean()

plt.figure(figsize=(12, 5))
sns.lineplot(x=success_by_hour.index, y=success_by_hour.values, marker='o', color='green')
plt.title("Taxa de Sucesso por Hora do Dia (Sucesso = Marcada ou >45s)")
plt.xlabel("Hora do Dia (São Paulo)")
plt.ylabel("Taxa de Sucesso (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(24))
plt.axhline(success_rate, color='red', linestyle='--', label=f'Média Global ({success_rate:.1%})')
plt.legend()
plt.savefig("artifacts/eda/tp_03_taxa_sucesso_hora.png")
plt.close()

# %%
# ── [SEÇÃO 7] Heatmap Temporal ───────────────────────────────────────────────────────────

pivot_tp = df_tp.pivot_table(index='dia_da_semana', columns='hora_do_dia', values='is_success', aggfunc='mean')
pivot_tp.index = [dias_label[i] for i in pivot_tp.index]

plt.figure(figsize=(15, 8))
sns.heatmap(pivot_tp * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Taxa de Sucesso (%)'})
plt.title("Mapa de Calor: Probabilidade de Sucesso por Dia e Hora (%)")
plt.xlabel("Hora do Dia (São Paulo)")
plt.ylabel("Dia da Semana")
plt.savefig("artifacts/eda/tp_05_heatmap_temporal.png")
plt.close()

# %%
# ── [SEÇÃO 8] Análise de Correlação das Features ──────────────────────────────────────────

features_corr = [
    'n_tentativas', 'horas_desde_primeiro_contato', 'horas_desde_ultimo_contato', 
    'densidade_tentativas', 'pressao_recente', 'hora_do_dia', 'ls_probabilidade', 
    'tentativas_mesma_hora', 'tentativas_mesmo_dia', 
    'mesma_hora_que_ultimo_contato', 'is_success'
]

corr_matrix = df_tp[features_corr].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title("Matriz de Correlação das Features (Timing Predict V2)")
plt.savefig("artifacts/eda/tp_06_correlacao.png")
plt.close()

print("\n--- MATRIZ DE CORRELAÇÃO ---")
print(corr_matrix['is_success'].sort_values(ascending=False))

# Exibir estatísticas descritivas das novas features
print("\n--- ESTATÍSTICAS DAS FEATURES DE FADIGA ---")
print(df_tp[['densidade_tentativas', 'pressao_recente']].describe())

# %%
# ── [SEÇÃO 9] Exportação para Treinamento ────────────────────────────────────────────────

TRAIN_SET_PATH = os.path.join("data", "timing_predict_train_set.csv")
print(f"\n4. Exportando dataset final para: {TRAIN_SET_PATH}")

# Selecionar as colunas incluindo as novas features de comportamento individual
cols_to_export = [
    'agent_id', 'created_at_brt', 'n_tentativas', 'horas_desde_primeiro_contato', 
    'horas_desde_ultimo_contato', 'densidade_tentativas', 'pressao_recente', 
    'ddd', 'ls_probabilidade', 'is_success',
    'tentativas_mesma_hora', 'tentativas_mesmo_dia', 
    'mesma_hora_que_ultimo_contato', 'last_outcome', 'primeira_reason'
]

df_tp[cols_to_export].to_csv(TRAIN_SET_PATH, index=False)
print(f"Exportação concluída! Total de linhas: {df_tp.shape[0]}")

# %%
df_tp.columns
# %%
