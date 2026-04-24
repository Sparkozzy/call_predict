# Metadados de Machine Learning — Call Predict

> **Objetivo:** Armazenar as categorias e parâmetros exatos usados no treinamento dos modelos XGBoost para garantir inferências idênticas ao ambiente de treino.

---

## 1. Categorias Globais (Compartilhadas)

### `ddd`
O modelo espera uma `string`. Inclui DDDs brasileiros + 'na'.
**Categorias:** `['11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '24', '27', '28', '29', '31', '32', '33', '34', '35', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48', '49', '51', '53', '54', '59', '61', '62', '63', '64', '65', '66', '67', '68', '69', '71', '73', '74', '75', '77', '79', '81', '82', '83', '84', '85', '86', '87', '88', '89', '91', '92', '94', '95', '97', '98', '99', 'na']`

---

## 2. Modelo: Lead Scoring (LS)
**Versão:** 1.0.0
**Arquivo:** `models/xgboost_LS_model.pkl`

### `Regiao` (Estado)
**Categorias Confirmadas:** `['AC', 'AL', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP']`

### `ultima_disconnection_reason`
Razão da última chamada.
**Categorias Probáveis (baseadas no treino):** `['agent_hangup', 'dial_busy', 'dial_failed', 'dial_no_answer', 'error_asr', 'error_retell', 'inactivity', 'invalid_destination', 'ivr_reached', 'max_duration_reached', 'primeiro_contato', 'telephony_provider_permission_denied', 'telephony_provider_unavailable', 'user_declined', 'user_hangup', 'voicemail_reached']`

---

## 3. Modelo: Timing Predict (TP)
**Versão:** 1.0.0
**Arquivo:** `models/xgboost_model_TP_V1.pkl`

### `dia_semana` (Inteiro como Categoria)
**Categorias:** `[0, 1, 2, 3, 4, 5, 6]`
**Tipo:** `int64` (Deve ser convertido via `.astype(int).astype('category')`)

### `hora_contato` (Inteiro)
Horário da predição (0-23).

---

## 4. Mapeamento de DDD para Região (Referência)

| DDD | Estado | DDD | Estado | DDD | Estado | DDD | Estado |
|---|---|---|---|---|---|---|---|
| 11-19 | SP | 21-24 | RJ | 27-28 | ES | 31-38 | MG |
| 41-46 | PR | 47-49 | SC | 51-55 | RS | 61 | DF |
| 62, 64 | GO | 63 | TO | 65-66 | MT | 67 | MS |
| 68 | AC | 69 | RO | 71-77 | BA | 79 | SE |
| 81, 87 | PE | 82 | AL | 83 | PB | 84 | RN |
| 85, 88 | CE | 86, 89 | PI | 91, 93, 94 | PA | 92, 97 | AM |
| 95 | RR | 96 | AP | 98, 99 | MA | | |
