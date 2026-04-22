import logging
from typing import Any, Dict
import os
from database import supabase
from utils import update_workflow_status, run_step_with_retry

logger = logging.getLogger(__name__)

async def process_call_predict(ctx: Dict[Any, Any], payload_dict: Dict[str, Any]):
    """
    Worker task: Executa a cascata de ML para predição de ligação com rastreabilidade completa.
    """
    # Extrair execution_id do job_id (formato: "job_<execution_id>")
    job_id = ctx.get("job_id", "")
    execution_id = job_id.replace("job_", "") if job_id.startswith("job_") else job_id
    
    logger.info(f"Iniciando process_call_predict ({execution_id})")

    # 1. Início de Execução: Atualizar status para RUNNING
    await update_workflow_status(execution_id, status="RUNNING")

    try:
        # Step: Validar Modelos no contexto
        async def check_models():
            model_ls = ctx.get("model_ls")
            model_tp = ctx.get("model_tp")
            if not model_ls or not model_tp:
                raise Exception("Modelos XGBoost não encontrados no contexto do Worker.")
            return "Modelos carregados"

        await run_step_with_retry(
            execution_id=execution_id,
            step_name="pre_call_processing_check_models",
            worker_func=check_models
        )

        # Step: Simulação de ETL (Próxima funcionalidade)
        async def etl_placeholder():
            # TODO: Implementar busca de features no Supabase
            return {"features": "mock"}

        await run_step_with_retry(
            execution_id=execution_id,
            step_name="pre_call_processing_etl_features",
            worker_func=etl_placeholder
        )

        # Finalização: SUCCESS
        await update_workflow_status(
            execution_id, 
            status="SUCCESS", 
            output_data={"status": "Workflow processado até o placeholder de ML"}
        )

    except Exception as e:
        logger.error(f"Erro fatal no workflow {execution_id}: {e}")
        await update_workflow_status(
            execution_id, 
            status="FAILED", 
            error_details=str(e)
        )
        return {"status": "failed", "error": str(e)}

    return {"status": "success", "execution_id": execution_id}
