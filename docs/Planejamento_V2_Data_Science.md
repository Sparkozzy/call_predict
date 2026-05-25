# Planejamento de Ciência de Dados: Modelos V2 (Mindflow)

Este documento estabelece o roadmap completo para a segunda versão (V2) dos modelos de predição de chamadas (Lead Scoring e Timing Predict). O objetivo é evoluir da fase exploratória via Jupyter Notebooks para uma esteira robusta, rastreável e modular, pronta para o ambiente de MLOps.

---

## 1. Arquitetura Proposta para V2

Com base na análise da V1, as features temporais isoladas tiveram pouco poder preditivo no modelo geral. A recomendação da V2 é adotar um **Dual-Model Approach**:

1.  **Modelo 1: Lead Scoring (Go/No-Go)**
    *   **Objetivo:** Determinar se o lead possui alta probabilidade de conversão/engajamento ao longo do seu ciclo de vida, independentemente da hora.
    *   **Saída:** Probabilidade (0 a 1) e recomendação ("Continuar Tentativas" vs "Descartar Lead").
2.  **Modelo 2: Timing Predict (Agendamento)**
    *   **Objetivo:** Dado que um lead passou no Modelo 1 (Go) e foi aprovado, determinar qual é o dia da semana e janela de horário ideal para contato. **Este modelo será treinado exclusivamente com o subconjunto de leads qualificados pelo Modelo 1.**
    *   **Saída:** Janela temporal sugerida (ex: Terça-feira, 14h-16h).

### 1.1. Grupo de Exploração (Controle)
A tabela `model_executions` contém registros que passam sem filtro de ML (`EXPLORAÇÃO`). Este grupo é vital para a saúde do projeto e deve ser tratado da seguinte forma:

*   **Análise (Baseline):** Ele serve como o nosso "Nível Zero". Qualquer melhoria (Lift) do modelo deve ser medida contra este grupo para provar que a IA é de fato superior ao envio aleatório.
*   **ETL (Flagging):** Implementar uma flag `is_exploration` no pipeline de dados para permitir a separação clara entre dados enviesados pelo modelo e dados de controle.
*   **Treinamento (Unbiased Data):** O grupo de exploração é a fonte de dados mais valiosa para o retreinamento, pois não sofre de "viés de seleção" (Selection Bias). Ele permite que o modelo aprenda com leads que seriam originalmente descartados pela versão anterior da IA.
*   **Peso Estratégico (Sample Weight):** Recomenda-se atribuir um **peso maior** (ex: 1.5x a 2x) para os registros do grupo de exploração durante o treinamento. Isso obriga o modelo a priorizar o aprendizado sobre a distribuição real da população em vez de apenas reforçar o viés das predições passadas da IA.

---

## 2. Fase 1: Investigação e Definição da Variável Alvo (Target)

Na V1, a métrica de sucesso pareceu atrelada a uma duração longa da chamada. Para a V2, precisaremos testar e comparar diferentes *targets* para entender qual melhor traduz o sucesso do negócio.

### Variáveis Alvo Candidatas:
*   **Target A (Duração Limiar):** Chamadas com Duração > X segundos (ex: > 45s ou > 60s). Representa leads que passaram do "Hook" (engajamento inicial).
*   **Target B (Razão de Desconexão):** Baseado na coluna `disconnection_reason`. Identificação explícita de sucesso x falha pela plataforma (ex: ignorar 'voicemail', focar em 'user_hangup' após um tempo razoável).
*   **Target C (Conversão Real/Marcada):** Focar em ligações onde a variável `marcada = true`. Investigar se esta variável é o melhor indicador de sucesso e quais features mais se correlacionam com ela.
*   **Target D (Híbrido - Score Ponderado com Penalização):** Criar um score que valoriza conversões/duração mas **punindo** (atribuindo valor negativo ou zero) ligações que atendem mas terminam em < 15 segundos ou terminam por inatividade.
*   **Target E (Híbrido de Engajamento):** Criar um target contínuo (Regressão) combinando duração, extração de dados e análise de sentimentos/intenções (caso existam transcrições).

### Como Compararemos os Targets:
Durante a EDA, validaremos cada *target* respondendo:
1.  **Balanceamento:** O *target* gera uma classe minoritária muito extrema (< 1%)?
3.  **Valor de Negócio:** Prever este *target* reduz custo com voicemail ou aumenta agendamentos reais?
4.  **Análise de Relevância (Marcada):** Avaliar se a variável `marcada` faz sentido analítico como target principal ou se é ruidosa demais para o treinamento, identificando quais variáveis estão mais correlacionadas a este evento.

---

## 3. Fase 2: Pipeline de ETL Modulada e Adaptável

O fluxo de ETL deixará de ser um Notebook estático e deverá ser transformado em classes ou scripts modulares. Assim, ao introduzirmos novas variáveis no futuro, o *core* de limpeza permanece intacto.

### Etapa 2.1: Processos Padrão Indispensáveis (Limpeza Base)
Independentemente de quais *features* ou *targets* formos testar na V2, o *DataFrame* inicial sempre passará pelas seguintes transformações de sanitização:
1.  **Deduplicação:** Remoção de eventuais registros duplicados na extração (baseado no ID único da ligação).
2.  **Correção de Tipagem e Unidades:** Conversão imperativa das colunas de duração (que vêm da API em milissegundos `ms`) para segundos (`s`) ou minutos.
3.  **Alinhamento Temporal (Fuso Horário):** 
    *   Conversão dos *timestamps* de UTC para o fuso brasileiro (`America/Sao_Paulo` - BRT), garantindo que as extrações de 'hora do dia' reflitam a realidade operacional.
4.  **Tratamento de Nulos Operacionais:** 
    *   Tratamento de nulos lógicos (ex: `disconnection_reason` = nulo significa o quê no contexto do Supabase?).
    *   Preenchimento ou descarte de instâncias defeituosas.

### Etapa 2.2: Engenharia de Features (Expansível)
Após a base limpa, aplicamos as transformações dependentes de modelo:
*   **Temporal:** Extração aprimorada (Hora, Dia da Semana, Mês, Feriados).
*   **Histórico de Engajamento (Anti-Leakage):** Quantidade de tentativas anteriores, tempo desde a última tentativa. **Nota: Durações anteriores (média, max, last) foram removidas para evitar data leakage e focar no primeiro sucesso.**
*   **Encoding:** One-Hot Encoding ou Target Encoding para variáveis categóricas (DDD, Provedor da Lista, Status).
*   **Feature Selection:** Descarte dinâmico de IDs de lead, colunas redundantes, e colunas com *Data Leakage* (informações que só saberemos *depois* da ligação terminar).

---

## 4. Fase 3: Aprofundamento Analítico via EDA

Antes de rodar o treinamento, executaremos as seguintes frentes usando a skill *EDA Agent*:

1.  **Análise de "Hook" / Sobrevivência:**
    *   A maior queda ocorre nas tentativas 1 para 2. Qual o limite prático de insistência antes do lead virar puro custo? Calcular a Taxa de Retenção x Nº de Tentativas.
2.  **Acurácia da Identificação de Máquina:**
    *   Avaliar se a IA consegue diferenciar *Voicemail* de um humano nos primeiros 10 segundos para reduzir custos inúteis.
3.  **Correlação Cruzada de Sazonalidade:**
    *   Em vez de olhar apenas para "A taxa é maior às 14h?", cruzar isso com o dia da semana e com a *tentativa de contato*. (Ex: "A 3ª tentativa funciona melhor na sexta de manhã?").

---

## 5. Fase 4: Treinamento, MLOps e Validação

1.  **Pipeline ML:** Construir pipelines no `scikit-learn` (`Pipeline`, `ColumnTransformer`) contendo os passos de escalonamento (`StandardScaler`) e encoding. Isso evita *Data Leakage* na hora do *Cross-Validation*.
2.  **Balanceamento:** Continuar utilizando técnicas de pesos (como `scale_pos_weight` do XGBoost) ou métodos de reamostragem (SMOTE) para lidar com a escassez do Target.
3.  **MLflow Tracking:** Todo experimento V2 (teste do Target A vs Target B, XGBoost vs Random Forest, tuning via GridSearchCV) será logado no servidor MLflow, salvando hiperparâmetros, F1-Score/ROC-AUC e os artefatos do modelo treinado.
4.  **Seleção e Deploy:** Os modelos selecionados serão exportados para consumo futuro do n8n/Supabase.
