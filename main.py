import os
import uuid
from fastapi import FastAPI, HTTPException, status
from arq import create_pool
from arq.connections import RedisSettings
from dotenv import load_dotenv

from schemas import CallPredictPayload, AcceptedResponse
from utils import register_workflow_execution

load_dotenv()

app = FastAPI(
    title="Call Predict API - Recepcionista",
    description="Endpoint para receber eventos de leads e enfileirar predições de ML.",
    version="1.0.0"
)

# Tratamento da URL do Redis no Easypanel
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
if "@" in REDIS_URL:
    protocol_user_pass, host_port = REDIS_URL.rsplit("@", 1)
    protocol_user_pass = protocol_user_pass.replace("#", "%23")
    REDIS_URL = f"{protocol_user_pass}@{host_port}"

# Pool do Redis será inicializado no startup da API
redis_pool = None

@app.on_event("startup")
async def startup_event():
    global redis_pool
    redis_pool = await create_pool(RedisSettings.from_dsn(REDIS_URL))

@app.on_event("shutdown")
async def shutdown_event():
    if redis_pool:
        await redis_pool.close()

@app.post(
    "/webhook/predict",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AcceptedResponse,
    tags=["Webhook"]
)
async def predict_webhook(payload: CallPredictPayload):
    """
    Recebe os dados do lead e enfileira a tarefa de predição no Worker.
    
    Retorna 202 Accepted imediatamente.
    """
    execution_id = str(uuid.uuid4())
    
    # Início do Fluxo: Registrar entrada em workflow_executions como PENDING
    await register_workflow_execution(
        execution_id=execution_id,
        workflow_name="pre_call_processing",
        input_data=payload.model_dump()
    )
    
    try:
        # Enfileira a tarefa no ARQ (Redis)
        await redis_pool.enqueue_job(
            "process_call_predict",
            payload.model_dump(),
            _job_id=f"job_{execution_id}"
        )
        
        return AcceptedResponse(
            message="Tarefa call_predict enfileirada com sucesso.",
            execution_id=execution_id
        )
        
    except Exception as e:
        # Erros de conexão com Redis, etc.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao enfileirar tarefa: {str(e)}"
        )

@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {"status": "ok", "redis": "connected" if redis_pool else "disconnected"}
