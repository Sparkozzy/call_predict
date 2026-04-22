import logging
from typing import Any, Dict
from arq import create_pool
from arq.connections import RedisSettings
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

async def process_call_predict(ctx: Dict[Any, Any], payload: Dict[str, Any]):
    """
    Worker task: Executa a cascata de ML para predição de ligação.
    
    Por enquanto, apenas confirma o recebimento e o acesso aos modelos no contexto.
    """
    logger.info(f"Iniciando process_call_predict para payload: {payload}")
    
    # Verifica se os modelos estão no contexto (injetados pelo startup do worker)
    model_ls = ctx.get("model_ls")
    model_tp = ctx.get("model_tp")
    
    if model_ls and model_tp:
        logger.info("Modelos XGBoost carregados com sucesso no contexto do Worker.")
    else:
        logger.warning("Atenção: Modelos XGBoost não encontrados no contexto do Worker.")

    # TODO: Implementar lógica de ETL e Cascata na próxima etapa
    logger.info(f"Tarefa process_call_predict recebida com sucesso. Número: {payload.get('numero')}")
    
    return {"status": "success", "message": "Task received and logged"}
