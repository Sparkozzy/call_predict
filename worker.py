import os
import logging
import xgboost as xgb
from arq.connections import RedisSettings
from dotenv import load_dotenv
from services import process_call_predict

# Configuração de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

async def startup(ctx):
    """
    Hook de inicialização do Worker.
    Carrega os modelos XGBoost para a RAM uma única vez.
    """
    logger.info("Iniciando Worker: Carregando modelos XGBoost para a memória...")
    
    # Caminhos dos modelos
    ls_model_path = "models/lead_scoring.json"
    tp_model_path = "models/timing_predict.json"
    
    # Carregamento Lead Scoring
    try:
        if os.path.exists(ls_model_path):
            model_ls = xgb.Booster()
            model_ls.load_model(ls_model_path)
            ctx["model_ls"] = model_ls
            logger.info(f"Modelo Lead Scoring carregado: {ls_model_path}")
        else:
            logger.error(f"Arquivo de modelo não encontrado: {ls_model_path}")
            ctx["model_ls"] = None
    except Exception as e:
        logger.error(f"Erro ao carregar modelo Lead Scoring: {e}")
        ctx["model_ls"] = None

    # Carregamento Timing Predict
    try:
        if os.path.exists(tp_model_path):
            model_tp = xgb.Booster()
            model_tp.load_model(tp_model_path)
            ctx["model_tp"] = model_tp
            logger.info(f"Modelo Timing Predict carregado: {tp_model_path}")
        else:
            logger.error(f"Arquivo de modelo não encontrado: {tp_model_path}")
            ctx["model_tp"] = None
    except Exception as e:
        logger.error(f"Erro ao carregar modelo Timing Predict: {e}")
        ctx["model_tp"] = None

async def shutdown(ctx):
    """Hook de encerramento do Worker."""
    logger.info("Encerrando Worker...")

class WorkerSettings:
    """Configurações do Worker ARQ."""
    functions = [process_call_predict]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
