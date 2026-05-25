# %% [markdown]
# # Feature Engineering — Lead Scoring V2 (Pandas Pro)
# 
# ## Objetivo
# Reescrito utilizando as melhores práticas do pandas-pro skill.
# Foco em vetorização, method chaining, e otimização de memória.

# %%
import pandas as pd
import numpy as np
import os
import sys
import io

if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# %%
# ── CARREGAMENTO E OTIMIZAÇÃO ────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "Retell_calls_V2_Final.csv")

print(f"Carregando e otimizando dados de: {DATA_PATH}...")

# Carregamento inicial de uma pequena amostra para analisar dtypes pode ser feito, 
# mas iremos definir os cruciais como strings para evitar perdas
dtypes = {'to_number': str, 'from_number': str, 'call_id': str}
df = pd.read_csv(DATA_PATH, low_memory=False, dtype=dtypes)

# %%
# ── LIMPEZA VETORIZADA ────────────────────────────────────────────────────────
df['created_at'] = pd.to_datetime(df['created_at'])

df = (
    df.sort_values('created_at', ascending=True)
    .dropna(subset=['disconnection_reason'])
    .drop_duplicates(subset=['call_id'], keep='last')
)

# Filtro de outliers (<= 35 tentativas) usando transform para eficiência
attempt_counts = df.groupby('to_number', observed=True)['call_id'].transform('count')
df = df.loc[attempt_counts <= 35].copy()

# %%
# ── ENGENHARIA DE FEATURES ───────────────────────────────────────────────────

# Sucesso real
df['real_call_success'] = (df['duration_s'] >= 60).astype(np.int8)

# Regra de Permeabilidade
df['target'] = df.groupby('to_number', observed=True)['real_call_success'].transform('max')

# Features Sequenciais
df = df.sort_values(['to_number', 'created_at'])

# Tentativa N
df['tentativa_n'] = df.groupby('to_number', observed=True).cumcount() + 1
df['tentativa_n'] = pd.to_numeric(df['tentativa_n'], downcast='integer')

# Motivos de Desconexão
df['last_reason'] = df.groupby('to_number', observed=True)['disconnection_reason'].shift(1).fillna('first_call')
primeira_reason = df.groupby('to_number', observed=True)['disconnection_reason'].transform('first')
df['primeira_reason'] = np.where(df['tentativa_n'] == 1, 'first_call', primeira_reason)

# Tempo
primeira_chamada = df.groupby('to_number', observed=True)['created_at'].transform('first')
df['horas_desde_primeiro_contato'] = (df['created_at'] - primeira_chamada).dt.total_seconds() / 3600.0
df['horas_desde_primeiro_contato'] = pd.to_numeric(df['horas_desde_primeiro_contato'], downcast='float')

# DDD
df['ddd'] = df['to_number'].str.replace(r'\.0$', '', regex=True).str.extract(r'^55(\d{2})').fillna('desconhecido')

# %%
# ── FILTRO DE SUCESSO ÚNICO E DOWNCASTING ────────────────────────────────────

# Filtro
df['any_success_before'] = df.groupby('to_number', observed=True)['real_call_success'].shift(1).fillna(0)
df['any_success_so_far'] = df.groupby('to_number', observed=True)['any_success_before'].cummax()

df_model = df.loc[(df['any_success_so_far'] == 0) & (df['real_call_success'] == 0)].copy()

# Tratamento do last_outcome unificando last_disconnection_reason e last_reason (conforme o script original no treinamento)
if 'last_disconnection_reason' in df_model.columns:
    df_model['last_outcome'] = df_model['last_disconnection_reason'].fillna(df_model['last_reason'])
else:
    df_model['last_outcome'] = df_model['last_reason']

# Colunas a remover
cols_to_drop = [
    'call_id', 'agent_version', 'id', 'from_number', 'target_marcada', 
    'Marcada', 'created_at', 'last_duration_s', 'agent_name', 'target_conversion',
    'success_at_this_call', 'any_success_before', 'any_success_so_far', 
    'disconnection_reason', 'to_number', 'real_call_success',
    'created_at_brt', 'duration_s', 'last_disconnection_reason', 'last_reason', 'target_engagement'
]
df_model = df_model.drop(columns=cols_to_drop, errors='ignore')

# Conversão final de strings com baixa cardinalidade para categoricals
cat_cols = ['agent_id', 'last_outcome', 'primeira_reason', 'ddd']
for col in cat_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype('category')

print(f"Shape final: {df_model.shape}")
print(f"Memória usada: {df_model.memory_usage(deep=True).sum() / 1e6:.2f} MB")

SAVE_PATH = os.path.join("data", "LS_training_data_v2.csv")
df_model.to_csv(SAVE_PATH, index=False)
print(f"Salvo em {SAVE_PATH}")

# %%
df_model.columns
# %%
