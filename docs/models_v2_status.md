# Status do Modelo Lead Scoring V2

Este documento registra a transição para o **Lead Scoring V2** e define os parâmetros operacionais vigentes.

## 1. Configuração Atual
- **Modelo Selecionado:** V2 (Intensive Evolutionary XGBoost)
- **Threshold Operacional:** `0.4`
- **Data da Ativação:** 2026-05-15

## 2. Identificação do Modelo
- **Nome do Arquivo:** `best_intensive_model_v2.json`
- **Diretório de Armazenamento:** `models/`
- **Parâmetros Técnicos:** `models/best_intensive_params.json`
- **Experimento MLflow:** `lead_scoring_v2_intensive`

## 3. Performance do Modelo (Threshold 0.4)
Avaliação realizada em conjunto de teste hold-out (dados nunca vistos pelo modelo).

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **Recall** | 98.31% | Capacidade de capturar leads que realmente convertem. |
| **Falsos Negativos** | 4 | Leads que converteriam mas foram erroneamente bloqueados. |
| **Precision** | 5.46% | Qualidade das recomendações feitas pelo modelo. |
| **Falsos Positivos** | 4016 | Leads recomendados que não converteram (custo de discagem). |
| **F1-Score** | 10.35% | Equilíbrio entre precisão e recall. |
| **AUC ROC** | 0.8463 | Capacidade global de distinção entre as classes. |

## 4. Mudanças e Melhorias (V1 vs V2)

### 4.1 Robustez Metodológica
- **Split Triplo:** Diferente da V1, a V2 utiliza divisão entre **Treino, Validação e Teste**. Isso elimina o vazamento de dados (*Data Leakage*) que ocorria na V1 ao selecionar parâmetros baseando-se no conjunto de teste.
- **Validação Cega:** As decisões de evolução do modelo foram tomadas exclusivamente no conjunto de Validação, garantindo que o Teste final seja uma métrica honesta de generalização.

### 4.2 Engenharia de Dados (Pandas Pro)
- **Otimização de Memória:** Implementação de *downcasting* e tipos categóricos no pipeline de dados, reduzindo o consumo de RAM.
- **Vetorização:** Substituição de loops e operações custosas por métodos vetorizados do Pandas.
- **Unificação de Features:** A feature `last_outcome` (unificação de motivos técnicos e de negócio) agora é gerada diretamente no pipeline de dados, garantindo consistência entre treino e inferência.

### 4.3 Estratégia de Treinamento
- **Treinamento Intensivo:** Aumento do número de estimadores e uso de técnicas de regularização (`subsample` e `colsample_bytree`).
- **Early Stopping:** O modelo interrompe o treinamento automaticamente ao detectar que a performance na validação parou de melhorar, prevenindo *overfitting*.
- **Postura Punitiva:** O treinamento foi configurado para penalizar severamente Falsos Negativos, priorizando a segurança de não perder leads qualificados.

## 5. Próximos Passos
- Monitorar a taxa de conversão real em produção com o novo threshold de 0.4.
- Avaliar se o volume de discagem (Falsos Positivos) está dentro do orçamento operacional.
