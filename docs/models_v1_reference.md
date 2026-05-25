# Referência Técnica: Modelos V1 (XGBoost)

Este documento serve como base comparativa para o desenvolvimento da V2 dos modelos de Lead Scoring e Timing Predict.

## 1. Lead Scoring (LS) — `xgboost_LS_model.pkl`
**Objetivo:** Avaliar a probabilidade de um lead ter uma ligação bem-sucedida (> 90s) em algum momento.

### Features Utilizadas na V1:
- **Geografia:** DDD e Região (Estado) do lead.
- **Histórico de Tentativas:** Número total de tentativas anteriores e o tempo (em horas) desde o primeiro contato.
- **Comportamento Passado:** Contagem de motivos de desconexão anteriores (ocupado, caixa postal, recusada).
- **Último Status:** A razão exata da última desconexão (`ultima_disconnection_reason`).

### Lógica de Negócio:
- Retorna score de 0 a 1.
- Identifica "leads difíceis" para evitar desperdício de recursos.

---

## 2. Timing Predict (TP) — `xgboost_model_TP_V1.pkl`
**Objetivo:** Encontrar a janela de tempo ideal (0-23h) para a ligação nas próximas 24 horas.

### Features Utilizadas na V1:
- **Tempo:** Dia da semana e Hora do dia.
- **Cadência:** Densidade de tentativas (por hora) e "pressão recente" (proximidade do último contato).
- **Geografia:** DDD do lead.

### Lógica de Negócio:
- Simulação de hora em hora para as próximas 24h (dentro do horário comercial).
- Escolhe o horário com maior probabilidade de sucesso.

---

## 3. Fluxo Técnico e Governança
- **Exploração (Grupo de Controle):** 5% das chamadas ignoram o modelo para coletar dados imparciais (`is_exploration=True`).
- **Tratamento Categórico:** Categorias fixas para DDDs, Estados e Motivos de Erro.
- **Carregamento:** Via `joblib` no startup do worker (Singleton).

---

## 4. Gaps Identificados para V2
- **Agent_id:** Ausência do perfil do agente como feature preditora.
- **Adaptabilidade:** Necessidade de capturar padrões de comportamento de leads que variam conforme o agente que está ligando.
