# Arquitetura Técnica — Call Predict (EDW)

> **Fonte da Verdade Técnica.** Este documento descreve a infraestrutura e as decisões arquiteturais do projeto. Toda mudança estrutural deve refletir aqui antes de ser codificada.
> **Instância Prod:** `https://call-predict-github.bkpxmb.easypanel.host/`
> **Repo:** `https://github.com/Sparkozzy/call_predict.git`

---

## Visão Geral

O projeto `call_predict` é um sistema orientado a eventos (Event-Driven Workflow — EDW) que utiliza uma cascata de modelos de Machine Learning (XGBoost) para decidir **se** e **quando** ligar para um lead. A arquitetura é composta por duas camadas desacopladas:

| Camada | Papel | Tecnologia |
|---|---|---|
| **Recepcionista (API)** | Recebe eventos, valida payload, enfileira tarefa | FastAPI |
| **Worker (Processamento)** | Executa a lógica pesada de ML de forma assíncrona | ARQ + Redis |

```
[Cliente/Webhook] → POST /webhook/predict
       │
       ▼
┌─────────────────────────────────┐
│   FastAPI (Recepcionista)       │  ← Responde 202 Accepted imediatamente
│   Valida payload (Pydantic)     │
│   Enfileira job no Redis (ARQ)  │
└──────────────┬──────────────────┘
               │  Redis Queue
               ▼
┌─────────────────────────────────┐
│   ARQ Worker (Processamento)    │  ← Modelos XGBoost carregados na RAM
│   Executa process_call_predict  │
│   Cascata Lead Scoring → TP     │
│   Persiste em model_executions  │
└──────────────┬──────────────────┘
               │
               ▼
         [Supabase DB]
```

---

## Componentes

### 1. FastAPI — Recepcionista

- **Arquivo**: `main.py`
- **Responsabilidade**: Exclusivamente receber, validar e enfileirar. **Nunca** executa lógica de ML.
- **Regra**: Retorna `202 Accepted` em < 50ms. Não bloqueia.

### 2. Redis + ARQ — Fila de Tarefas

- **Papel**: Fila de mensagens persistente entre a API e o Worker.
- **Configuração**: definida em `worker.py` via `WorkerSettings`.
- **Garantia de Durabilidade**: Jobs sobrevivem a restarts do servidor enquanto o Redis estiver ativo.

### 3. ARQ Worker — Processador ML

- **Arquivo**: `worker.py`
- **Hook `startup`**: Carrega os modelos XGBoost **uma única vez** na inicialização do processo e os injeta no dicionário de contexto `ctx`. Isso evita I/O de disco a cada chamada.
- **Tarefa principal**: `process_call_predict(ctx, payload)` — definida em `services.py`.
- **Deduplicação**: O worker isola registros por `call_id` únicos antes da inferência, garantindo que as features de contagem reflitam chamadas reais e não eventos múltiplos da mesma ligação.

---

## Integração XGBoost (Padrão Singleton no Worker)

Os modelos são arquivos `.json` treinados externamente e salvos localmente. O carregamento segue o princípio **Singleton**: carrega-se uma vez no `startup` do worker e reutiliza-se a instância em memória para todas as inferências subsequentes.

```python
# Pseudocódigo do startup
async def startup(ctx):
    import xgboost as xgb
    ctx["model_ls"] = xgb.Booster()
    ctx["model_ls"].load_model("models/xgboost_LS_model.pkl")

    ctx["model_tp"] = xgb.Booster()
    ctx["model_tp"].load_model("models/xgboost_model_TP_V1.pkl")
```

**Por quê?** Carregar um modelo XGBoost do disco a cada inferência levaria ~200-500ms de I/O desnecessário. Com o Singleton no Worker, o custo de carregamento é pago **uma única vez** ao iniciar o processo.

---

## Banco de Dados (Supabase)

### Tabelas de Rastreabilidade (Padrão Mestre-Detalhe EDW)

| Tabela | Função |
|---|---|
| `workflow_executions` | Registro mestre de cada execução de workflow (PENDING → RUNNING → SUCCESS/FAILED) |
| `workflow_step_executions` | Registro detalhado de cada nó/step de um workflow |

### Tabela: `model_executions` (Nova — call_predict)

Registra **cada inferência** realizada pela cascata XGBoost. É a fonte de verdade para auditoria e retreinamento dos modelos.

| Coluna | Tipo | Restrição | Descrição |
|---|---|---|---|
| `id` | `uuid` | PK, `gen_random_uuid()` | Identificador único da inferência |
| `created_at` | `timestamptz` | NOT NULL | Timestamp UTC da criação |
| `model_id` | `text` | NOT NULL | Identificador do modelo (ex: `lead_scoring_v1`) |
| `model_name` | `text` | NOT NULL | Nome legível do modelo |
| `model_version` | `text` | NOT NULL | Versão semântica (ex: `1.0.0`) |
| `to_number` | `text` | NOT NULL | Número do lead avaliado |
| `agent_id` | `text` | NOT NULL | ID do agente Retell associado |
| `ls_probabilidade` | `float8` | nullable, CHECK [0,1] | Probabilidade bruta do Lead Scoring |
| `ls_decisao` | `text` | nullable, CHECK ('LIGAR','DESCARTAR') | Decisão do Lead Scoring |
| `tp_horario_escolhido` | `int4` | nullable, CHECK [0,23] | Hora UTC escolhida pelo Timing Predict |
| `tp_probabilidade_pico` | `float8` | nullable, CHECK [0,1] | Probabilidade do horário de pico |
| `is_exploration` | `bool` | NOT NULL | `true` se pertence ao grupo de controle (5%) |

> **Nota**: A tabela `model_executions` já existe no Supabase com as constraints de CHECK definidas. **Não recriar**.

---

## Variáveis de Ambiente

Todas as credenciais devem estar em `.env` na raiz do projeto (ignorado pelo git):

```env
SUPABASE_URL=...
SUPABASE_KEY=...
REDIS_URL=redis://localhost:6379
```

---

*Documento mantido pelo padrão EDW. Toda mudança de infra exige atualização aqui primeiro.*
