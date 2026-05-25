# Relatório Consolidado: EDA Retell Calls (Mindflow V2)

Este relatório resume os achados da Análise Exploratória de Dados realizada para fundamentar os modelos de **Lead Scoring** e **Timing Predict** na versão V2.

---

## 1. Qualidade e Sanitização dos Dados
*   **Deduplicação:** Foram removidos 60.557 registros duplicados (baseados em `id` único).
*   **Tratamento de Nulos:**
    *   `duration_s`: Nulos convertidos para **0**, pois correspondem a falhas de conexão.
    *   `call_id`: Nulos preenchidos com "call_123" para consistência de ID.
*   **Filtragem Crítica:** O agente "WhatsApp" foi excluído por apresentar comportamento ruidoso que prejudicaria o aprendizado do modelo.

## 2. Estrutura de Agentes (Consolidação)
Para ganhar volume estatístico, os agentes foram agrupados em 4 categorias principais:
1.  **Agente Mindflow Disparo:** Maior volume, focado em prospecção fria.
2.  **Agente Mindflow (Formulário):** Focado em Inbound.
3.  **agente_socios:** Fusão dos agentes de Infoprodutores, Imobiliárias e Reagendamento.
4.  **Agente Gatekeeper:** Focado em transpor barreiras iniciais.

## 3. Definição de Targets (Sucesso)
Dada a raridade do evento `Marcada` (apenas 74 casos em 107k), adotaremos um **Dual-Target Approach**:

| Target | Nome | Critério | Finalidade |
| :--- | :--- | :--- | :--- |
| **Target A** | Engajamento | `duration_s > 60` | Treinar o modelo a identificar leads que "ficam na linha". |
| **Target B** | Conversão | `Marcada == True` | Refinar o modelo para o fechamento real (Sparse Learning). |

## 4. Análise de Duração e Thresholds
*   **75% das chamadas** duram menos de **4 segundos**.
*   **95% das chamadas** duram menos de **42 segundos**.
*   **99% das chamadas** duram menos de **68 segundos**.
*   **Recomendação:** O limiar de 60 segundos é o "divisor de águas" entre uma chamada automatizada/ruído e um atendimento humano real.

## 5. Insights de Comportamento Sequencial (A "Regra de Ouro")
Esta foi a descoberta mais importante para o **Lead Scoring V2**:

### A. O Efeito da Tentativa
*   **1ª Tentativa:** Taxa de sucesso **0%**.
*   **2ª e 3ª Tentativas:** Pico de conversão (0,32%).
*   **Implicação:** O modelo deve ser treinado sabendo que a primeira ligação é um investimento de "abertura de porta".

### B. O Valor do Histórico (Sem Duração)
A probabilidade de sucesso em uma ligação é influenciada pela sequência, mas ignoraremos a duração das chamadas anteriores para evitar *Data Leakage* e focar no objetivo de **Sucesso Único**:
*   **Foco no Primeiro Sucesso:** Uma vez que o lead atinge o Target (A ou B), chamadas subsequentes são descartadas da análise, pois o objetivo do negócio é converter o lead uma única vez.
*   **Sequencialidade Pura:** O preditor principal passa a ser o número da tentativa (`tentativa_n`) e eventos categóricos (ex: `last_reason`), excluindo métricas de tempo passadas.

---

## 6. Próximos Passos (Roadmap de Modelagem)

1.  **Feature Engineering (Nova Estratégia):**
    *   **Exclusão de Duração Passada:** Removidas colunas `max_duration_ever` e `last_duration` para evitar vazamento de dados.
    *   Criar `tentativa_n`: O número sequencial da chamada.
    *   Criar `last_reason`: O motivo da última desconexão (categórico).
    *   **Filtragem de Sucesso Único:** Treinar o modelo apenas com dados até a primeira ocorrência de sucesso por lead.
2.  **Treinamento:**
    *   Iniciar com um Classificador XGBoost/LightGBM focado no **Target A (> 60s)**.
    *   Utilizar validação cruzada respeitando a linha do tempo (Time-Series Split).
3.  **Timing:**
    *   Analisar o intervalo de tempo (Gap) entre a 1ª e a 2ª tentativa para leads que tiveram `duration > 15s`.

---
**Data do Relatório:** 11/05/2026
**Analista:** Antigravity (EDA Agent)
