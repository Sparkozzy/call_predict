# %%
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Garantir diretório de artefatos
os.makedirs("artifacts/eda", exist_ok=True)

# 1. Carregamento
FILE_PATH = "data/Retell_calls_Mindflow.csv"
try:
    df = pd.read_csv(FILE_PATH)
    print(f"Shape inicial: {df.shape}")
    print("\nVisualização inicial:")
    print(df.head())
except Exception as e:
    print(f"Erro ao carregar arquivo: {e}")
    exit()

# 2. Limpeza Padrão Indispensável
# ms -> s
# Verificando nomes de colunas para duração
dur_col = next((c for c in df.columns if 'duration' in c.lower()), None)
if dur_col:
    print(f"Usando '{dur_col}' para conversão de duração.")
    df['duration_s'] = df[dur_col] / 1000.0
else:
    print("AVISO: Nenhuma coluna de duração encontrada.")

# Timestamp UTC -> BRT
time_cols = [c for c in df.columns if 'time' in c.lower() or 'at' in c.lower()]
print(f"Colunas de tempo detectadas: {time_cols}")

for col in time_cols:
    try:
        # Tenta converter para datetime e depois para BRT
        df[col] = pd.to_datetime(df[col], errors='coerce')
        # Se já tiver timezone, converte. Se não, localiza como UTC e converte.
        if df[col].dt.tz is not None:
            df[f"{col}_brt"] = df[col].dt.tz_convert('America/Sao_Paulo')
        else:
            df[f"{col}_brt"] = df[col].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
        print(f"Coluna '{col}' convertida para BRT em '{col}_brt'.")
    except Exception as e:
        print(f"Erro ao converter {col}: {e}")

# %%
# ── [SEÇÃO] Ajuste de Unidades e Qualidade de Dados ──────────────────────────

# 1. Correção da Duração (ms -> s)
if 'Duracao' in df.columns:
    print(f"Convertendo 'Duracao' para 'duration_s'...")
    df['duration_s'] = df['Duracao'] / 1000.0
    print(df['duration_s'].describe())
else:
    print("ERRO: Coluna 'Duracao' não encontrada.")

# 2. Remoção de Duplicatas (Indispensável conforme plano)
original_count = len(df)

# Identificar colunas críticas
id_col = next((c for c in df.columns if 'call_id' in c.lower() or 'id' in c.lower()), None)
# Usaremos a primeira coluna de tempo brt criada ou a original para ordenar
time_ref = next((c for c in df.columns if '_brt' in c.lower()), 
                next((c for c in df.columns if 'at' in c.lower() or 'time' in c.lower()), None))

if id_col and time_ref:
    print(f"Identificado ID: '{id_col}' e Referência Temporal: '{time_ref}'")
    
    # Contagem antes
    duplicates_count = df.duplicated(subset=[id_col]).sum()
    print(f"Total de registros duplicados (baseado em {id_col}): {duplicates_count}")
    
    if duplicates_count > 0:
        # Ordenar para garantir que o 'keep=last' pegue o mais recente se estiver em ordem crescente
        df = df.sort_values(by=time_ref, ascending=True)
        df = df.drop_duplicates(subset=[id_col], keep='last')
        
        print(f"Remoção concluída. Registros restantes: {len(df)}")
        print(f"Impacto: {original_count - len(df)} registros removidos.")
    else:
        print("Nenhuma duplicata encontrada.")
else:
    print(f"ERRO: Não foi possível identificar ID ({id_col}) ou Tempo ({time_ref}) para deduplicação.")

print("\nVisualização após deduplicação:")
print(df.head())

# %% 
# ── [SEÇÃO] Limpeza de Colunas e Análise de Redundância ───────────────────────

cols_to_drop = [
    'Nome', 'Email', 'data', 'Numero', 'call_type', 'transcript', 
    'recording_url', 'eleven_labs_cost', 'LLM', 'LLM_cost', 
    'combined_cost', 'call_summary', 'LLM_token_usage', 
    'segmento', 'equipe', 'data_brt'
]

# Remover apenas colunas que existem no DF
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_cols_to_drop)

print(f"Colunas removidas: {existing_cols_to_drop}")
print(f"Colunas restantes ({len(df.columns)}): {df.columns.tolist()}")

# Identificação de Redundâncias
# 1. Correlação para numéricas
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    corr_matrix = numeric_df.corr()
    print("\nMatriz de Correlação (Redundância Numérica):")
    print(corr_matrix)
    
    # Salvar heatmap de correlação
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
    plt.title("Heatmap de Correlação")
    plt.tight_layout()
    plt.savefig("artifacts/eda/01_correlacao_features.png")
    plt.close()

# 2. Verificar colunas com valor único (baixa variância)
single_value_cols = [c for c in df.columns if df[c].nunique() <= 1]
print(f"\nColunas com valor único ou constantes: {single_value_cols}")

print("\nVisualização final após limpeza:")
print(df.head())
# %% 
# ── [SEÇÃO] Refinamento da Limpeza e Consolidação ───────────────────────────

# 1. Comparação de IDs (Verificar se id e call_id são redundantes)
diffs = -1
if 'id' in df.columns and 'call_id' in df.columns:
    try:
        diffs = (df['id'].astype(str) != df['call_id'].astype(str)).sum()
        print(f"Diferenças entre 'id' e 'call_id': {diffs} registros de {len(df)}")
    except:
        pass

# 2. Remoção Final de Redundâncias e Colunas Inúteis
final_drop = ['Duracao', 'status', 'status_brt']
if diffs == 0:
    final_drop.append('id')
    print("Colunas 'id' e 'call_id' são idênticas. Removendo 'id'.")

df = df.drop(columns=[c for c in final_drop if c in df.columns])

# 3. Diagnóstico de Nulos Final
print("\nRelatório de Valores Nulos Final:")
print(df.isnull().sum())

# 4. Salvar Dataset Limpo
df.to_csv("data/Retell_calls_Cleaned.csv", index=False)
print(f"\nDataset consolidado salvo em 'data/Retell_calls_Cleaned.csv'. Shape: {df.shape}")

print("\nColunas Finais:")
print(df.columns.tolist())

# %% 
# ── [SEÇÃO] Tratamento de Nulos e Investigação de Desconexão ────────────────

# 1. Tratamentos Solicitados
df['duration_s'] = df['duration_s'].fillna(0)
df['call_id'] = df['call_id'].fillna('call_123')

print("Tratamentos aplicados: duration_s (NaN -> 0), call_id (NaN -> 'call_123')")

# 2. Investigação: disconnection_reason IS NULL
null_disc = df[df['disconnection_reason'].isnull()].copy()

print(f"\n--- Investigação de disconnection_reason (N= {len(null_disc)}) ---")

# A. Qual a duração média dessas chamadas?
print(f"Duração média quando motivo é nulo: {null_disc['duration_s'].mean():.2f}s")
print(f"Distribuição de duração (Motivo Nulo):\n{null_disc['duration_s'].value_counts().head()}")

# B. Elas ocorrem em algum período específico?
if 'created_at_brt' in df.columns:
    print(f"\nRange temporal dos nulos: {null_disc['created_at_brt'].min()} até {null_disc['created_at_brt'].max()}")

# C. Existe correlação com agent_name?
print(f"\nTop 5 Agentes com mais motivos nulos:\n{null_disc['agent_name'].value_counts().head()}")

# Visualização da distribuição de duração para nulos vs não-nulos
plt.figure(figsize=(10, 5))
plt.hist(df[df['disconnection_reason'].notnull()]['duration_s'], bins=50, alpha=0.5, label='Com Motivo', density=True)
plt.hist(df[df['disconnection_reason'].isnull()]['duration_s'], bins=50, alpha=0.5, label='Motivo Nulo', density=True)
plt.title("Distribuição de Duração: Motivo de Desconexão Nulo vs Presente")
plt.xlabel("Duração (s)")
plt.ylabel("Densidade")
plt.legend()
plt.savefig("artifacts/eda/02_investigacao_nulos_desconexao.png")
plt.close()

print("\nGráfico comparativo salvo em 'artifacts/eda/02_investigacao_nulos_desconexao.png'")


# %% 
# ── [SEÇÃO] Exclusão de Agentes Problemáticos e Refino da Fusão ─────────────

# 1. Remover Agente WhatsApp (identificado como problemático)
original_size = len(df)
# Filtramos antes de qualquer fusão para garantir que pegamos o nome original ou o já renomeado
df = df[~df['agent_name'].str.contains('whatsapp|agente_socios', na=False, case=False) | 
        ~df['agent_name'].str.contains('whatsapp', na=False, case=False)]
# Nota: Como o script roda do zero, o filtro abaixo é mais seguro:
df = df[~df['agent_name'].str.contains('whatsapp', na=False, case=False)]

print(f"Agente WhatsApp removido. Registros excluídos: {original_size - len(df)}")

# 2. Nova Fusão de Sócios (Sem WhatsApp)
termos_socios_final = 'Infoprodutores|Imobiliárias|reagendamento|agente_sócios'
mask_socios = df['agent_name'].str.contains(termos_socios_final, na=False, case=False)

mapping_check = df[mask_socios][['agent_name', 'agent_id']].drop_duplicates()

if not mapping_check.empty:
    new_agent_id = mapping_check['agent_id'].iloc[0]
    df.loc[mask_socios, 'agent_id'] = new_agent_id
    df.loc[mask_socios, 'agent_name'] = 'agente_socios'
    print(f"Fusão final para 'agente_socios' concluída (Sem WhatsApp).")
else:
    print("AVISO: Agentes para fusão não encontrados.")

# Verificação
print("\nDistribuição final de chamadas por agente:")
print(df['agent_name'].value_counts())

# %% 
# ── [SEÇÃO] Análise de Performance por Agente ────────────────────────────────

# 1. Agrupamento por Agente
agent_stats = df.groupby('agent_name').agg(
    total_chamadas=('id', 'count'),
    duracao_media=('duration_s', 'mean'),
    taxa_marcada_perc=('Marcada', lambda x: (x == True).sum() / len(x) * 100)
).sort_values(by='duracao_media', ascending=False)

print("Estatísticas por Agente Consolidadas:")
print(agent_stats)

# 2. Visualização Dual: Volume vs Duração Média
fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Agente')
ax1.set_ylabel('Duração Média (s)', color=color)
ax1.bar(agent_stats.index, agent_stats['duracao_media'], color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=15)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Volume de Chamadas', color=color)
ax2.plot(agent_stats.index, agent_stats['total_chamadas'], color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Performance por Agente: Duração Média vs Volume")
fig.tight_layout()
plt.savefig("artifacts/eda/03_performance_agentes.png")
plt.close()

# %% 
# ── [SEÇÃO] Análise de Correlação com o Target 'Marcada' ─────────────────────

# 1. Preparação do Target (Lidando com strings 'true'/'false' ou nulos)
def clean_marcada(val):
    if pd.isna(val): return 0
    if isinstance(val, str):
        return 1 if val.lower() == 'true' else 0
    return 1 if val is True else 0

df['target_marcada'] = df['Marcada'].apply(clean_marcada)

print(f"\nTotal de chamadas Marcadas (Sucesso): {df['target_marcada'].sum()}")

# 2. Correlação Numérica
# Filtramos apenas colunas que fazem sentido correlacionar
cols_analise = ['duration_s', 'agent_version', 'target_marcada']
correlations = df[cols_analise].corr()['target_marcada'].sort_values(ascending=False)
print("\nCorrelação com 'Marcada':")
print(correlations)

# 3. Padrão por Categoria (Onde estão as Marcadas?)
print("\nTaxa de 'Marcada' por Motivo de Desconexão (Onde houve sucesso):")
disc_stats = df.groupby('disconnection_reason').agg(
    total=('target_marcada', 'count'),
    marcadas=('target_marcada', 'sum'),
    taxa_sucesso=('target_marcada', 'mean')
).sort_values(by='taxa_sucesso', ascending=False)
print(disc_stats[disc_stats['marcadas'] > 0])

# 4. Cruzamento Duração vs Marcada
print("\nEstatísticas de Duração para chamadas Marcadas vs Não-Marcadas:")
duracao_stats = df.groupby('target_marcada')['duration_s'].describe()
print(duracao_stats)

# Visualização: Boxplot Duração vs Marcada
plt.figure(figsize=(10, 6))
# Usando dropna para garantir que o boxplot não quebre com nulos remanescentes
data_to_plot = [df[df['target_marcada'] == 0]['duration_s'].dropna(), 
                df[df['target_marcada'] == 1]['duration_s'].dropna()]
plt.boxplot(data_to_plot, labels=['Não Marcada', 'Marcada'])
plt.title("Distribuição de Duração por Status de Marcação")
plt.ylabel("Duração (s)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("artifacts/eda/04_boxplot_duracao_marcada.png")
plt.close()

print("\nVisualização salva em 'artifacts/eda/04_boxplot_duracao_marcada.png'")

# %% 
# ── [SEÇÃO] Distribuição Estatística de Duração ──────────────────────────────

print("\n--- Estatísticas Descritivas de Duração (Percentis Detalhados) ---")
# Percentis ajudam a definir o limiar de sucesso (Target A)
stats_duracao = df['duration_s'].describe(percentiles=[.25, .5, .75, .85, .9, .95, .99])
print(stats_duracao)

# Visualização Combinada (Boxplot + Histograma)
# Usamos o seaborn (sns) para uma estética premium
plt.figure(figsize=(12, 8))
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

ax_main = plt.subplot(grid[1:, :])
ax_top = plt.subplot(grid[0, :], sharex=ax_main)

# Boxplot no topo
sns.boxplot(x=df['duration_s'], ax=ax_top, color='skyblue', width=0.4)
ax_top.set(xlabel='', yticks=[], title='Distribuição de Duração das Chamadas (Geral)')

# Histograma/KDE embaixo
sns.histplot(df['duration_s'], ax=ax_main, kde=True, color='blue', bins=100)
ax_main.set_xlabel('Duração (segundos)')
ax_main.set_ylabel('Frequência (Escala Log)')
ax_main.set_yscale('log') # Escala log para ver a cauda longa (outliers)

# Linhas de referência
ax_main.axvline(df['duration_s'].mean(), color='red', linestyle='--', label=f'Média: {df["duration_s"].mean():.2f}s')
ax_main.axvline(df['duration_s'].median(), color='green', linestyle='-', label=f'Mediana: {df["duration_s"].median():.2f}s')
ax_main.legend()

plt.savefig("artifacts/eda/05_distribuicao_duracao.png")
plt.close()

print("\nGráfico de distribuição salvo em 'artifacts/eda/05_distribuicao_duracao.png'")

# %% 
# ── [SEÇÃO] Análise de Histórico e Prevenção de Leakage ──────────────────────

# 1. Contagem de Chamadas Longas (> 300s)
calls_300 = len(df[df['duration_s'] > 300])
print(f"\nChamadas com mais de 300s: {calls_300} ({calls_300/len(df)*100:.3f}% da base)")

# 2. Engenharia de Ordem de Tentativa (Cumulative Count)
# Ordenamos por to_number e tempo antes de contar para evitar leakage
df = df.sort_values(['to_number', 'created_at_brt'])
df['tentativa_n'] = df.groupby('to_number').cumcount() + 1

print("\nDistribuição de Volume por Número de Tentativa (Top 10):")
print(df['tentativa_n'].value_counts().head(10))

# 3. Taxa de Sucesso (Marcada) por Tentativa
tentativa_stats = df.groupby('tentativa_n').agg(
    total=('target_marcada', 'count'),
    marcadas=('target_marcada', 'sum'),
    taxa_sucesso=('target_marcada', 'mean')
).head(10)

print("\nPerformance (Taxa de Marcada) por Ordem de Tentativa:")
print(tentativa_stats)

# Visualização
plt.figure(figsize=(10, 6))
sns.barplot(x=tentativa_stats.index, y=tentativa_stats['taxa_sucesso'] * 100, color='orange', alpha=0.8)
plt.title("Taxa de Conversão vs Ordem de Tentativa")
plt.xlabel("Número da Tentativa (Ligação nº X para o mesmo Lead)")
plt.ylabel("Taxa de Marcada (%)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig("artifacts/eda/06_sucesso_por_tentativa.png")
plt.close()

print("\nVisualização salva em 'artifacts/eda/06_sucesso_por_tentativa.png'")

# %% 
# ── [SEÇÃO] Distribuição Acumulada e Definição de Threshold ──────────────────

# 1. Cálculo da ECDF (Empirical Cumulative Distribution Function)
durations = df['duration_s'].dropna()
sorted_dur = np.sort(durations)
yvals = np.arange(len(sorted_dur)) / float(len(sorted_dur) - 1)

plt.figure(figsize=(10, 6))
plt.plot(sorted_dur, yvals, marker='.', linestyle='none', color='blue', alpha=0.5)
plt.title("Distribuição Acumulada (ECDF) da Duração")
plt.xlabel("Duração (segundos)")
plt.ylabel("Probabilidade Acumulada P(Duração <= X)")
plt.grid(True, linestyle='--', alpha=0.7)

# Destacar percentis chave no gráfico
plt.axhline(0.95, color='orange', linestyle='--', label='95% das chamadas')
plt.axhline(0.99, color='red', linestyle='--', label='99% das chamadas')
plt.legend()

plt.savefig("artifacts/eda/07_ecdf_duracao.png")
plt.close()

# 2. Boxplot por Agente Consolidado (Escala Log)
plt.figure(figsize=(12, 7))
sns.boxplot(x='agent_name', y='duration_s', data=df, hue='agent_name', palette='viridis', legend=False)
plt.yscale('log')
plt.title("Comparativo de Retenção por Agente (Escala Log)")
plt.xlabel("Agente")
plt.ylabel("Duração (s) - Escala Log")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.savefig("artifacts/eda/08_boxplot_agentes_log.png")
plt.close()

# 3. Recomendação Final de Threshold
threshold_95 = df['duration_s'].quantile(0.95)
threshold_99 = df['duration_s'].quantile(0.99)

print("\n--- RECOMENDAÇÃO PARA TARGET A (ENGAJAMENTO) ---")
print(f"Percentil 95%: {threshold_95:.2f}s")
print(f"Percentil 99%: {threshold_99:.2f}s")
print(f"Sugestão: Definir Sucesso como Duração > 60 segundos.")
print("Isso garante uma amostra de ~1% da base para o modelo aprender padrões de retenção real.")

print("\nArtefatos de fechamento salvos em 'artifacts/eda/'")

# %% 
# ── [SEÇÃO] Análise Sequencial: Impacto do Passado no Sucesso Futuro ──────────

# 1. Engenharia de Features de Histórico (Lagged Features)
# Trazemos o motivo e a duração da ligação ANTERIOR para a linha da ligação atual
df = df.sort_values(['to_number', 'created_at_brt'])
df['last_disconnection_reason'] = df.groupby('to_number')['disconnection_reason'].shift(1)
df['last_duration_s'] = df.groupby('to_number')['duration_s'].shift(1)

# 2. Filtrar apenas chamadas que têm um passado (n > 1)
df_hist = df[df['tentativa_n'] > 1].copy()

print(f"\nAnalisando {len(df_hist)} chamadas que possuem histórico anterior para o mesmo número.")

# 3. Agrupamento: Como o motivo da última ligação afeta a chance de sucesso nesta?
seq_analysis = df_hist.groupby('last_disconnection_reason').agg(
    total_tentativas_seguintes=('target_marcada', 'count'),
    sucessos_nesta_tentativa=('target_marcada', 'sum'),
    taxa_sucesso_futuro=('target_marcada', 'mean')
).sort_values(by='taxa_sucesso_futuro', ascending=False)

print("\n--- Influência do Motivo de Desconexão Anterior no Sucesso Atual ---")
# Mostramos apenas motivos com volume relevante (> 50 ocorrências)
relevant_seq = seq_analysis[seq_analysis['total_tentativas_seguintes'] > 50]
print(relevant_seq)

# 4. Visualização: Probabilidade Condicional
plt.figure(figsize=(12, 6))
sns.barplot(
    x=relevant_seq.index, 
    y=relevant_seq['taxa_sucesso_futuro'] * 100, 
    hue=relevant_seq.index,
    palette='coolwarm',
    legend=False
)
plt.title("Taxa de Conversão Atual vs. Motivo da Ligação Anterior")
plt.xlabel("Motivo da Desconexão na Ligação ANTERIOR")
plt.ylabel("Probabilidade de Sucesso (%)")
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("artifacts/eda/09_analise_sequencial_motivos.png")
plt.close()

print("\nVisualização salva em 'artifacts/eda/09_analise_sequencial_motivos.png'")

# 5. Insight de Duração Passada
# Será que ligações passadas mais longas aumentam a chance de conversão na próxima?
df_hist['last_duration_bin'] = pd.cut(df_hist['last_duration_s'], bins=[0, 5, 15, 30, 60, 1000], labels=['0-5s', '5-15s', '15-30s', '30-60s', '60s+'])
dur_seq = df_hist.groupby('last_duration_bin', observed=True)['target_marcada'].mean() * 100

print("\n--- Taxa de Sucesso Atual vs Duração da Ligação Anterior ---")
print(dur_seq)

# %% 
# ── [SEÇÃO] Exportação Final do Dataset Tratado ──────────────────────────────

# Definindo o nome do arquivo final para o treinamento V2
final_output_path = 'data/Retell_calls_V2_Final.csv'

# Salvando o DataFrame com todas as transformações, fusões de agentes e features sequenciais
df.to_csv(final_output_path, index=False)

print(f"\n[OK] DATASET MESTRE (V2) EXPORTADO!")
print(f"Local: {final_output_path}")
print(f"Shape: {df.shape}")
print(f"Features de Histórico incluídas: ['tentativa_n', 'last_disconnection_reason', 'last_duration_s']")


# %%
