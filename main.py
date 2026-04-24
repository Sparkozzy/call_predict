import os
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.responses import JSONResponse
from arq import create_pool
from arq.connections import RedisSettings
from dotenv import load_dotenv

from schemas import PredictWebhookInput
from supabase import create_client, Client

# Carregar variáveis de ambiente
load_dotenv()

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Call Predict API", version="1.0.0")

# --- CLIENTES ---
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")  # Alterado para bater com o .env
)

async def get_redis():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    return await create_pool(RedisSettings.from_dsn(redis_url))

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(ZoneInfo("UTC")).isoformat()}

@app.post("/webhook/predict", status_code=status.HTTP_202_ACCEPTED)
async def webhook_predict(
    payload: PredictWebhookInput,
    x_api_key: str = Header(None),
    redis = Depends(get_redis)
):
    """
    Nó 1: Recebe o lead, valida o formato e enfileira para processamento.
    """
    # 1. Validação de API Key (opcional, dependendo da sua segurança)
    # if x_api_key != os.getenv("WEBHOOK_API_KEY"):
    #     raise HTTPException(status_code=401, detail="Chave de API inválida")

    # 2. Validação de Negócio: Número deve começar com '+'
    if not payload.numero.startswith("+"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O número de telefone deve estar no formato internacional (+55...)"
        )

    try:
        # 3. Criação do Registro Mestre em workflow_executions (Rastreabilidade EDW)
        execution_res = supabase.table("workflow_executions").insert({
            "workflow_name": "call_predict",
            "status": "PENDING",
            "input_data": payload.dict(),
            "created_at": datetime.now(ZoneInfo("UTC")).isoformat()
        }).execute()
        
        execution_id = execution_res.data[0]["id"]

        # 4. Enfileiramento para o Worker (Redis/ARQ)
        await redis.enqueue_job(
            "process_call_predict_task",
            data=payload.dict(),
            execution_id=execution_id
        )

        logger.info(f"Lead {payload.numero} enfileirado com sucesso. Exec ID: {execution_id}")

        return {
            "status": "Accepted",
            "execution_id": execution_id,
            "message": "Lead enfileirado para predição"
        }

    except Exception as e:
        logger.error(f"Erro ao enfileirar lead: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar webhook"
        )
