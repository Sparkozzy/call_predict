# %% [markdown]
# # Feature Engineering — Lead Scoring V2
# 
# ## Objetivo
# Este arquivo realiza o processamento e transformação dos dados históricos de ligações (Retell) em um conjunto de features 
# otimizado para o modelo de **Lead Scoring V2 (Go/No-Go)**. 
# 
# O objetivo central é criar preditores robustos que identifiquem a probabilidade de um lead se engajar 
# ou converter, utilizando a Regra de Permeabilidade:
# - Target 1: Se o lead teve AO MENOS UMA ligação >= 60s, TODAS as suas tentativas são Target 1.
# 
# Focando especialmente em:
# 1. **Comportamento Sequencial:** Criar histórico de tentativas (`tentativa_n`).
# 2. **Anti-Leakage Strategy:** Ignorar totalmente durações de chamadas passadas.
# 3. **Sucesso Único:** Filtrar a jornada até o primeiro sucesso definido pela regra híbrida.
# 4. **Definição de Targets:** Consolidar alvos de Engajamento Híbrido e Conversão.
# 5. **Sanitização Final:** Preparar o dataset para treinamento.

# %%
import pandas as pd
import numpy as np
import os
import sys
import io

# Fix para emojis/UTF-8 no console do Windows (conforme dados.md)
# Ajustado para ser compatível com ambientes interativos (Jupyter/OutStream)
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# %%
# ── CARREGAMENTO DOS DADOS ─────────────────────────────────────────────────────

DATA_PATH = os.path.join("data", "Retell_calls_V2_Final.csv")

print(f"Carregando dados de: {DATA_PATH}...")

# Usando dtype={'to_number': str} para evitar perda de precisão e notação científica (Float)
df = pd.read_csv(DATA_PATH, low_memory=False, dtype={'to_number': str, 'from_number': str, 'call_id': str})

print(f"Dataset carregado com sucesso! Shape: {df.shape}")

# Inspeção inicial obrigatória (conforme EDA-agent e dados.md)
print("\nPrimeiras 5 linhas:")
print(df.head())

print("\nTipos de colunas:")
print(df.dtypes)

# %%
# ── LIMPEZA DE DUPLICATAS E ONGOING ───────────────────────────────────────────

# Erro Crasso detectado: O dataset original contém múltiplas linhas para o mesmo call_id (ongoing e ended).
# Precisamos garantir que cada call_id seja ÚNICO e que seja o registro FINAL (com resultado).

print(f"\nIniciando limpeza de duplicatas por call_id (Lógica Blindada)...")
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at', ascending=True)

# 1. Remover PRIMEIRO o que é inconclusivo (ongoing/null)
# Isso garante que se houver um 'ended' e um 'ongoing', ficaremos com o 'ended' para a deduplicação.
total_antes_null = len(df)
df = df.dropna(subset=['disconnection_reason'])
print(f"Chamadas inconclusivas (ongoing/null) descartadas: {total_antes_null - len(df)}")

# 2. Agora deduplicamos para manter apenas o registro FINAL de cada call_id
total_antes_dup = len(df)
df = df.drop_duplicates(subset=['call_id'], keep='last')
print(f"Duplicatas de call_id removidas: {total_antes_dup - len(df)}")
print(f"Dataset sanitizado e blindado! Novo shape: {df.shape}")

# %%
# ── FILTRO DE OUTLIERS ─────────────────────────────────────────────────────────

# Removendo leads com mais de 35 tentativas (conforme análise estatística do Boxplot)
# Isso evita que o modelo aprenda com anomalias de sistema ou números 'zumbis'.
print(f"\nFiltrando leads com mais de 35 tentativas...")
total_leads_inicial = df['to_number'].nunique()

lead_counts = df['to_number'].value_counts()
leads_validos = lead_counts[lead_counts <= 35].index
df = df[df['to_number'].isin(leads_validos)]

total_leads_final = df['to_number'].nunique()
print(f"Filtro aplicado! Leads removidos: {total_leads_inicial - total_leads_final}")
print(f"Novo shape do dataset: {df.shape}")

# %%
# ── ENGENHARIA DE FEATURES: TARGETS ──────────────────────────────────────────

# Conforme Decisão de Design (docs/Plano_Features_LeadScoring.md):
# Target A (Engajamento Híbrido): 
#   - 1 se Duração >= 60s
#   - 1 se Duração [30-59s] E motivo for 'user_hangup'
# Target B (Conversão): Marcada == True

# 1. Sucesso Real da Chamada (Apenas > 60s conforme solicitado)
df['real_call_success'] = (df['duration_s'] >= 60).astype(int)

# 2. Target de Engajamento Propagado (Regra de Permeabilidade)
# Se o lead teve sucesso (>=60s) em qualquer momento, marcamos todas as tentativas como 1.
df['target_engagement'] = df.groupby('to_number')['real_call_success'].transform('max')

# Garantindo que Marcada seja booleano/inteiro
# Algumas vezes vem como string 'True'/'False' ou nulo
df['Marcada'] = df['Marcada'].fillna(False)
if df['Marcada'].dtype == 'object':
    df['target_conversion'] = df['Marcada'].apply(lambda x: str(x).lower() == 'true').astype(int)
else:
    df['target_conversion'] = df['Marcada'].astype(int)

print(f"Distribuição Target Engagement: {df['target_engagement'].value_counts(normalize=True).to_dict()}")
print(f"Distribuição Target Conversion: {df['target_conversion'].value_counts(normalize=True).to_dict()}")

# %%
# ── ENGENHARIA DE FEATURES: VARIÁVEIS SEQUENCIAIS (ANTI-LEAKAGE) ───────────────

# Para garantir a ordem correta das tentativas, precisamos ordenar por lead e tempo
print("\nCalculando variáveis sequenciais por lead (to_number)...")

df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values(['to_number', 'created_at'])

# 1. Tentativa Número (tentativa_n)
df['tentativa_n'] = df.groupby('to_number').cumcount() + 1

# 2. Motivo da Última Desconexão (last_reason) - Categórico, sem duração
df['last_reason'] = df.groupby('to_number')['disconnection_reason'].shift(1).fillna('first_call')

# 3. Motivo da Primeira Desconexão (primeira_reason)
# O motivo da tentativa 1. Para evitar data leakage na própria tentativa 1, usamos 'first_call'.
primeira_reason_real = df.groupby('to_number')['disconnection_reason'].transform('first')
df['primeira_reason'] = np.where(df['tentativa_n'] == 1, 'first_call', primeira_reason_real)

# 4. Horas desde o Primeiro Contato
primeira_chamada_time = df.groupby('to_number')['created_at'].transform('first')
df['horas_desde_primeiro_contato'] = (df['created_at'] - primeira_chamada_time).dt.total_seconds() / 3600.0

# 5. DDD (Extração do to_number)
# Limpamos qualquer '.0' residual e extraímos os 2 dígitos após o '55' (DDI Brasil)
df['ddd'] = df['to_number'].str.replace(r'\.0$', '', regex=True).str.extract(r'^55(\d{2})')
df['ddd'] = df['ddd'].fillna('desconhecido')

# 6. FILTRO DE SUCESSO ÚNICO
# Identificar se já houve sucesso (Target A ou B) em chamadas anteriores para o mesmo lead
print("Aplicando filtro de Sucesso Único (removendo chamadas pós-conversão)...")

# Usamos o real_call_success para o filtro, para preservar a jornada ATÉ o sucesso
df['success_at_this_call'] = df['real_call_success']
df['any_success_before'] = df.groupby('to_number')['success_at_this_call'].shift(1).fillna(0)
df['any_success_so_far'] = df.groupby('to_number')['any_success_before'].cummax()

# Filtramos: mantemos apenas chamadas onde não houve sucesso anterior
# E excluímos a própria chamada de sucesso (treinando o modelo EXCLUSIVAMENTE com o histórico de falhas pré-sucesso)
df_filtered = df[(df['any_success_so_far'] == 0) & (df['real_call_success'] == 0)].copy()

print(f"Dataset filtrado! De {len(df)} para {len(df_filtered)} linhas.")
print(f"Sucessos preservados: {df_filtered['success_at_this_call'].sum()}")

print("\nExemplo da jornada de um lead (to_number):")
cols_view = ['to_number', 'tentativa_n', 'horas_desde_primeiro_contato', 'ddd', 'primeira_reason', 'last_reason', 'target_engagement']
print(df_filtered[cols_view].head(10))

# %%
# ── PRÓXIMOS PASSOS ──────────────────────────────────────────────────────────
# 1. Encoding de variáveis categóricas (agent_name, last_reason, DDD).
# 2. Remover colunas auxiliares (success_at_this_call, any_success_before, etc).
# 3. Integração com MLflow.

# %%
# ── LIMPEZA FINAL DO DATASET PARA MODELAGEM ──────────────────────────────────

# Criando a cópia conforme solicitado
# Usaremos a df_filtered como base, pois ela já contém o filtro de "Sucesso Único"
df_model = df_filtered.copy()

cols_to_drop = [
    'call_id', 
    'agent_version', 
    'id', 
    'from_number', 
    'target_marcada', 
    'Marcada', 
    'created_at', 
    'last_duration_s',
    'agent_name',
    'target_conversion'
]

# Removendo também as colunas auxiliares e colunas específicas solicitadas para não poluir o modelo
cols_to_drop += [
    'success_at_this_call', 'any_success_before', 'any_success_so_far', 
    'disconnection_reason', 'to_number', 'real_call_success',
    'created_at_brt', 'duration_s'
]

print(f"Removendo colunas: {cols_to_drop}")
df_model = df_model.drop(columns=cols_to_drop, errors='ignore')

# Renomeando o target para o nome simplificado solicitado
df_model = df_model.rename(columns={'target_engagement': 'target'})

print(f"\nShape final para modelagem: {df_model.shape}")
print("\nColunas restantes (Features + Targets):")
print(df_model.columns.tolist())

print("\nPrimeiras linhas do dataset pronto:")
df_model.head()

# %%
# ── DISTRIBUIÇÃO DO TARGET NO DATASET FINAL ──────────────────────────────────

print("\n--- DISTRIBUIÇÃO DO TARGET (Engagement > 60s) ---")
counts = df_model['target'].value_counts()
percent = df_model['target'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Quantidade': counts,
    'Porcentagem (%)': percent
})

print(dist_df)

# Exibindo o desbalanceamento
if 1 in counts:
    imbalance_ratio = counts[0] / counts[1]
    print(f"\nRatio (Classe 0 / Classe 1): {imbalance_ratio:.2f} : 1")
else:
    print("\nAlerta: Nenhuma classe positiva (1) encontrada no dataset filtrado!")

# %%
# ── EXPORTAÇÃO DO DATASET FINAL ─────────────────────────────────────────────

SAVE_PATH = os.path.join("data", "LS_training_data.csv")
print(f"\nSalvando dataset de treinamento em: {SAVE_PATH}...")
df_model.to_csv(SAVE_PATH, index=False)
print("Arquivo salvo com sucesso!")

# %%
