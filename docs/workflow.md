# Workflow: `call_predict`

> **Objetivo**: Dado um lead (número de telefone + agent_id), decidir **se** vale a pena ligar (Lead Scoring) e **quando** ligar (Timing Predict) usando uma cascata de dois modelos XGBoost.

---

## Visão Geral da Cascata

```
[Webhook: POST /webhook/predict]
        │
        │  payload: { numero, agent_id }
        ▼
┌─────────────────────────────────────────────┐
│  ETAPA 0 — Sorteio: Grupo de Controle?      │
│  random() < 0.05 → is_exploration = True    │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┘
        │
   is_exploration?
   ├── TRUE  (5%)  → Pular ML, marcar como exploration
   │                  Ligar no horário padrão → LIGAR
   │
   └── FALSE (95%) ─────────────────────────────────────┐
                                                         ▼
                                          ┌──────────────────────────┐
                                          │  ETAPA 1 — ETL Features  │
                                          │  Buscar dados do lead    │
                                          │  no Supabase/histórico   │
                                          │  Montar pandas.DataFrame │
                                          └──────────┬───────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────────┐
                                          │  ETAPA 2 — Lead Scoring  │
                                          │  model_ls.predict(DMatrix)│
                                          │  ls_probabilidade → float │
                                          │  threshold > 0.5?         │
                                          │   ├── DESCARTAR → stop    │
                                          │   └── LIGAR → próximo     │
                                          └──────────┬───────────────┘
                                                     │ ls_decisao = LIGAR
                                                     ▼
                                          ┌──────────────────────────┐
                                          │  ETAPA 3 — Timing Predict│
                                          │  model_tp.predict(DMatrix)│
                                          │  Probabilidade por hora  │
                                          │  tp_horario_escolhido    │
                                          │  tp_probabilidade_pico   │
                                          └──────────┬───────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────────┐
                                          │  ETAPA 4 — Persistência  │
                                          │  INSERT model_executions  │
                                          │  Enfileirar ligação Retell│
                                          └──────────────────────────┘
```

---

## Regra Crítica: Grupo de Controle (5% Exploration)

**Motivação**: Para que os modelos não entrem em modo de "câmara de eco" (só ligam para quem eles mesmos previram como bom), 5% das execuções são sorteadas aleatoriamente como **grupo de controle**.

**Comportamento**:
- `is_exploration = True`.
- O ML é **completamente ignorado** (nenhuma inferência é feita).
- A ligação é feita para o lead, independentemente do score.
- O horário usado é o padrão do negócio (a definir).
- O resultado é registrado em `model_executions` com `ls_probabilidade = null`, `ls_decisao = null`.

**Implementação**:
```python
import random
is_exploration = random.random() < 0.05
```

**Por quê isso importa**: Os dados do grupo de controle alimentarão o retreinamento dos modelos com exemplos não-enviesados pela própria predição.

---

## Engenharia de Features (ETL) e Preparação de Dados

O workflow de preparação de dados deve ser desenhado para agir sobre o lead alvo, recuperando seu histórico e gerando o DataFrame final que alimentará os dois modelos. Ele é subdividido logicamente em duas etapas:

1. **Nó `get_rows_supabase`**:
   Busca o histórico de ligações do cliente na tabela `retell_calls_mindflow`.
   - **Filtro**: Traz apenas as linhas referentes ao `numero` específico.
   - **Limite**: Pega no máximo as 100 linhas mais recentes.

2. **Nó `Data_transform`**:
   Processa o histórico e garante a não-existência de dados nulos usando `.fillna(0)`.
   Os dados são separados e as *features* formatadas especificamente para cada modelo.

### Features: Modelo de Lead Scoring (LS)
```python
[
    # Informações Geográficas
    'ddd', 'Regiao',
    # Métricas de Tentativas
    'n_tentativas_anteriores', 'horas_desde_primeiro_contato',
    # Contagem de Razões de Disconexão/Status (Anteriores)
    'n_voicemail_reached_anteriores', 'n_dial_no_answer_anteriores',
    'n_inactivity_anteriores', 'N_invalid_destination_anteriores',
    'N_user_hangup_anteriores', 'N_user_declined_anteriores',
    'N_telephony_provider_permission_denied_anteriores', 'N_dial_busy_anteriores',
    'N_telephony_provider_unavailable_anteriores', 'N_agent_hangup_anteriores',
    'N_error_asr_anteriores', 'N_error_retell_anteriores',
    'N_dial_failed_anteriores', 'N_max_duration_reached_anteriores',
    'N_ivr_reached_anteriores',
    # Status da Última Interação
    'ultima_disconnection_reason'
]
```

### Features: Modelo de Timing Predict (TP)
```python
[
    'ddd',
    'hora_do_contato',
    'dia_da_semana',
    'hora_ultimo_contato',
    'densidade_tentativas',
    'pressao_recente'
]
```

### Guia de Cálculo Mínimo Funcional (Mentalidade Pandas)
*Como os dados foram preparados nas etapas de treinamento. Ao processarmos um único `numero` no Workflow (ordenado cronologicamente), a agregação usa a mesma lógica:*

**1. DDD e Região:**
```python
# 1. Extração
def get_ddd(phone):
    phone = str(phone).replace('+', '').replace('55', '')
    if not phone or len(phone) < 2: return np.nan
    return phone[:2]

# 2. Mapeamento
ddd_to_state = {
    '11': 'SP', '12': 'SP', '13': 'SP', '14': 'SP', '15': 'SP', '16': 'SP', '17': 'SP', '18': 'SP', '19': 'SP',
    '21': 'RJ', '22': 'RJ', '24': 'RJ', '27': 'ES', '28': 'ES', '31': 'MG', '32': 'MG', '33': 'MG', '34': 'MG',
    '35': 'MG', '37': 'MG', '38': 'MG', '41': 'PR', '42': 'PR', '43': 'PR', '44': 'PR', '45': 'PR', '46': 'PR',
    '47': 'SC', '48': 'SC', '49': 'SC', '51': 'RS', '53': 'RS', '54': 'RS', '55': 'RS', '61': 'DF', '62': 'GO',
    '63': 'TO', '64': 'GO', '65': 'MT', '66': 'MT', '67': 'MS', '68': 'AC', '69': 'RO', '71': 'BA', '73': 'BA',
    '74': 'BA', '75': 'BA', '77': 'BA', '79': 'SE', '81': 'PE', '82': 'AL', '83': 'PB', '84': 'RN', '85': 'CE',
    '86': 'PI', '87': 'PE', '88': 'CE', '89': 'PI', '91': 'PA', '92': 'AM', '93': 'PA', '94': 'PA', '95': 'RR',
    '96': 'AP', '97': 'AM', '98': 'MA', '99': 'MA'
}
df['Regiao'] = df['ddd'].map(ddd_to_state)
```

**2. Conversão Temporal:**
```python
df['created_at'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert('America/Sao_Paulo')
df['hora_contato'] = df['created_at'].dt.hour
df['dia_semana'] = df['created_at'].dt.dayofweek
```

**3. Histórico e Retrospectiva (com `shift` para prevenir *Data Leakage*):**
Para todos os cálculos que envolvem `_anteriores`, baseamo-nos no histórico até o ponto do evento.
```python
# Motivos de Desconexão Anteriores
reasons = [
    'voicemail_reached', 'dial_no_answer', 'inactivity', 'invalid_destination',
    'user_hangup', 'user_declined', 'telephony_provider_permission_denied',
    'dial_busy', 'telephony_provider_unavailable', 'agent_hangup', 'error_asr',
    'error_retell', 'dial_failed', 'max_duration_reached', 'ivr_reached'
]

for reason in reasons:
    col_name = f'n_{reason}_anteriores' if reason in ['voicemail_reached', 'dial_no_answer', 'inactivity'] else f'N_{reason}_anteriores'
    is_reason = (df['disconnection_reason'] == reason).astype(int)
    # Acumulado deslocado 1 posição p/ não contabilizar a si próprio
    df[col_name] = is_reason.cumsum().shift(1).fillna(0)

# Último status retornado no BD
df['ultima_disconnection_reason'] = df['disconnection_reason'].shift(1).fillna('primeiro_contato')
```

**4. Fadiga e Pressão de Resposta:**
```python
df['n_tentativas_anteriores'] = df.index  # (assumindo range index pós histórico ordenado)
df['horas_desde_primeiro_contato'] = (df['created_at'] - df['created_at'].iloc[0]).dt.total_seconds() / 3600
df['horas_desde_ultimo_contato'] = (df['created_at'] - df['created_at'].shift(1)).dt.total_seconds() / 3600

df['hora_ultimo_contato'] = df['hora_contato'].shift(1).fillna(-1)

df['densidade_tentativas'] = (df['n_tentativas_anteriores'] / (df['horas_desde_primeiro_contato'] + 1)).round(4)
df['pressao_recente'] = (df['n_tentativas_anteriores'] / (df['horas_desde_ultimo_contato'].fillna(0) + 1)).round(4)
```

---

## Steps do Workflow (`workflow_step_executions`)

Seguindo a convenção `call_predict_<OQF>`:

| Step Name | Descrição | Nó |
|---|---|---|
| `call_predict_enqueue` | Registro do job no Redis + criação do master em `workflow_executions` | API (síncrono) |
| `call_predict_sorteio_exploration` | Sorteio dos 5%; define `is_exploration` | Worker |
| `call_predict_etl_features` | Busca dados do lead e constrói `pandas.DataFrame` para inferência | Worker |
| `call_predict_lead_scoring` | Inferência do modelo Lead Scoring; produz `ls_probabilidade` e `ls_decisao` | Worker |
| `call_predict_timing_predict` | Inferência do modelo Timing Predict; produz `tp_horario_escolhido` e `tp_probabilidade_pico` | Worker |
| `call_predict_persist_model` | INSERT em `model_executions` com todos os resultados da cascata | Worker |
| `call_predict_trigger_call` | Enfileira a ligação Retell para o horário definido (via ARQ `_defer_until`) | Worker |

---

## Payload de Entrada

```json
{
  "numero": "+5511999999999",
  "agent_id": "agent_abc123"
}
```

**Validações (Pydantic)**:
- `numero`: string, obrigatório. Deve começar com `+` (validação de negócio).
- `agent_id`: string, obrigatório.

---

## Payload de Saída (202 Accepted — API)

```json
{
  "status": "accepted",
  "message": "Tarefa call_predict enfileirada.",
  "execution_id": "uuid-gerado"
}
```

---

## Modelos XGBoost

| Variável no ctx | Arquivo | Responsabilidade |
|---|---|---|
| `ctx["model_ls"]` | `models/xgboost_LS_model.pkl` | Lead Scoring — decide LIGAR ou DESCARTAR |
| `ctx["model_tp"]` | `models/xgboost_model_TP_V1.pkl` | Timing Predict — decide o melhor horário (0-23h) |

---

## Status do Workflow

- [x] **Passo 0** — Documentação criada
- [x] **Passo 1** — Dependências adicionadas
- [x] **Passo 2** — Schemas Pydantic criados
- [x] **Passo 3** — Endpoint `POST /webhook/predict` criado
- [x] **Passo 4** — Worker com startup de modelos + assinatura `process_call_predict`
- [ ] **Passo 5** — ETL de features (próxima sessão)
- [ ] **Passo 6** — Cascata Lead Scoring
- [ ] **Passo 7** — Timing Predict
- [ ] **Passo 8** — Persistência em `model_executions`
- [ ] **Passo 9** — Disparo da ligação Retell


## Server and deploy

O servidor está configurado para realizar deploy automático do código na branch main do github.

Link do repositório: https://github.com/Sparkozzy/call_predict.git

Para testar workflow em produção, execute comandos como no exemplo de requisição.

---

*Referência: `docs/architecture.md` · `docs/conventions.md`*
