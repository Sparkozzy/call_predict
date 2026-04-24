import os
import logging
import joblib
from arq.connections import RedisSettings
from dotenv import load_dotenv
from services import process_call_predict
from schemas import PredictWebhookInput

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Tratamento obrigatório da URL do Redis no Easypanel
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
if "@" in REDIS_URL:
    protocol_user_pass, host_port = REDIS_URL.rsplit("@", 1)
    protocol_user_pass = protocol_user_pass.replace("#", "%23")
    REDIS_URL = f"{protocol_user_pass}@{host_port}"

async def startup(ctx):
    """Hook de inicialização: Carrega os modelos .pkl para a RAM."""
    logger.info("Iniciando Worker: Carregando modelos .pkl para a memória...")
    
    ls_model_path = "models/xgboost_LS_model.pkl"
    tp_model_path = "models/xgboost_model_TP_V1.pkl"
    
    try:
        if os.path.exists(ls_model_path):
            ctx["model_ls"] = joblib.load(ls_model_path)
            logger.info(f"Modelo Lead Scoring carregado: {ls_model_path}")
        else:
            logger.error(f"Arquivo não encontrado: {ls_model_path}")
    except Exception as e:
        logger.error(f"Erro ao carregar Lead Scoring: {e}")

    try:
        if os.path.exists(tp_model_path):
            ctx["model_tp"] = joblib.load(tp_model_path)
            logger.info(f"Modelo Timing Predict carregado: {tp_model_path}")
        else:
            logger.error(f"Arquivo não encontrado: {tp_model_path}")
    except Exception as e:
        logger.error(f"Erro ao carregar Timing Predict: {e}")

async def shutdown(ctx):
    logger.info("Encerrando Worker...")

# --- TASK WRAPPERS ---

async def process_call_predict_task(ctx, data: dict, execution_id: str):
    """
    Wrapper para converter o dict do Redis de volta para Pydantic 
    e chamar o orquestrador.
    """
    payload = PredictWebhookInput(**data)
    await process_call_predict(ctx, payload, execution_id)

class WorkerSettings:
    functions = [process_call_predict_task]
    on_startup = startup
    on_shutdown = shutdown
    
    # Inicialização obrigatória via DSN
    redis_settings = RedisSettings.from_dsn(REDIS_URL)
    
    max_jobs = 50
    job_timeout = 300
