# Documentação de Feature Engineering: Lead Scoring V2

Este documento descreve o processamento, transformação de dados e a engenharia de features realizada para gerar o dataset de treinamento do modelo **Lead Scoring V2 (Go/No-Go)**. Todas as lógicas aplicadas respeitam a regra de **Anti-Leakage** (só utilizam informações disponíveis *antes* da chamada iniciar).

---

## 🎯 Definição do Target e Lógica de Permeabilidade

A definição do "sucesso" (target) evoluiu de uma visão pontual por chamada para uma visão focada no **potencial do lead**.

### 1. Sucesso Real da Chamada (A Regra dos 60s)
Uma ligação individual é considerada sucesso absoluto se a sua duração foi igual ou superior a 60 segundos (`duration_s >= 60`).

### 2. Regra de Permeabilidade (Propagação por Lead)
Como o objetivo do Lead Scoring é prever o **potencial do lead** engajar (mesmo que ele precise de várias tentativas para atender), aplicamos a Regra de Permeabilidade:
* **Se o lead obteve sucesso em ALGUMA tentativa, TODAS as suas tentativas são marcadas como `target_engagement = 1`.**
* Isso ensina o modelo a identificar leads valiosos em seus estágios iniciais, focando no perfil geral do lead e não apenas no evento isolado daquela chamada específica.

---

## 🧹 Preparação do Dataset e Filtros

Para garantir um treinamento limpo, livre de ruídos e data leakage, o script aplica os seguintes filtros:

1. **Filtro de Outliers (Limite de 35 Tentativas):** 
   * Análises estatísticas mostraram que a esmagadora maioria das operações se concentra em leads com até 30-35 tentativas.
   * Leads anômalos com centenas ou milhares de chamadas (zumbis/erros de sistema) foram totalmente excluídos para não distorcer as distribuições estatísticas do modelo.
2. **Filtro de Sucesso Único (Jornada):**
   * O modelo deve prever a probabilidade *até* o lead engajar.
   * Ordenamos a jornada cronologicamente e cortamos todas as chamadas **após a primeira ligação de sucesso real**. Se um lead engajou na 5ª tentativa, as tentativas de 1 a 5 entram no treino (com target=1), e da tentativa 6 em diante, todas são descartadas (evitando prever para quem já engajou).

**Resultado do pipeline:** Um dataset final robusto de **~75.430 linhas**, com um balanceamento saudável de **~21:1** (aproximadamente 4.5% de exemplos positivos).

---

## 🏗️ Features Criadas para o Modelo

As variáveis preditivas calculadas sequencialmente por lead (`to_number`), ordenadas pela data da chamada (`created_at`):

| Feature | Tipo | Lógica de Criação / Cálculo | Hipótese de Negócio |
| :--- | :--- | :--- | :--- |
| **`tentativa_n`** | Numérica | Contagem sequencial cumulativa por `to_number` (`cumcount + 1`). | A taxa de sucesso provavelmente decai após o N-ésimo contato. |
| **`ddd`** | Categórica | Extraído dos primeiros 2 caracteres da string `to_number`. | Regiões diferentes possuem padrões e horários de disponibilidade distintos. |
| **`last_reason`** | Categórica | Deslocamento (`shift(1)`) da razão de queda (`disconnection_reason`) da ligação anterior. É nula na primeira ligação. | Indica a "temperatura" da última interação (ex: *user_hangup* vs *voicemail*). |
| **`primeira_reason`** | Categórica | Extraída obrigatoriamente da 1ª chamada do lead e propagada para toda a jornada do mesmo. | A primeira reação técnica da linha (caixa postal, número inválido) prevê o comportamento futuro. |
| **`horas_desde_primeiro_contato`** | Numérica | Diferença entre a data atual e a data da 1ª tentativa, convertida para horas. | Mede a "idade" daquele lead em nossa base de contatos. Leads frescos convertem mais. |

---

## 🛠️ Próximos Passos: Como Alimentar o Modelo

O arquivo `data/LS_training_data.csv` já contempla todas as transformações acima. Para iniciar a modelagem, deve-se seguir:

1. **Limpeza Base:** Remover colunas de identificação e metadados irrelevantes para o modelo (ex: `to_number`, `created_at_brt`, `agent_id`, `real_call_success`). O Target final que deve ser passado ao estimador é o `target_engagement`.
2. **Imputação de Valores:** Features como `last_reason` possuem nulos na primeira tentativa (já que não há chamada anterior). Recomenda-se preenchê-los com uma constante (ex: `"first_call"` ou `"none"`).
3. **Encoding de Categóricas:** Transformar `ddd`, `last_reason` e `primeira_reason` em representações numéricas usando *One-Hot Encoding* ou *Target Encoding* (atenção para calcular o Target Encoding apenas nas partições de *treino* no K-Fold).
4. **Desbalanceamento (Scale Pos Weight):** Para a proporção de 21:1, configurar o classificador (XGBoost ou LightGBM) para compensar a classe minoritária. No XGBoost, ajusta-se `scale_pos_weight = (count(0) / count(1))`.
5. **Rastreabilidade:** Utilizar o MLflow para documentar e salvar todas as iterações (runs), armazenando os parâmetros do modelo e a matriz de confusão gerada.
