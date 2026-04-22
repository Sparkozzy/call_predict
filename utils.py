import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import zoneinfo

from database import supabase

logger = logging.getLogger(__name__)

# Fuso horário padrão de Brasília
BR_TZ = zoneinfo.ZoneInfo("America/Sao_Paulo")

def get_utc_now() -> str:
    """Retorna timestamp ISO 8601 em UTC para o banco."""
    return datetime.now(timezone.utc).isoformat()

def get_br_now() -> datetime:
    """Retorna datetime no fuso de Brasília para lógica interna."""
    return datetime.now(BR_TZ)

async def register_workflow_execution(
    execution_id: str,
    workflow_name: str,
    input_data: Dict[str, Any],
    status: str = "PENDING"
):
    """Registra ou atualiza a entrada mestre em workflow_executions."""
    data = {
        "id": execution_id,
        "workflow_name": workflow_name,
        "status": status,
        "input_data": input_data,
        "started_at": get_utc_now() if status == "RUNNING" else None,
        "created_at": get_utc_now()
    }
    # Upsert baseado no ID (execution_id)
    supabase.table("workflow_executions").upsert(data).execute()
    logger.info(f"Workflow {workflow_name} ({execution_id}) registrado como {status}.")

async def update_workflow_status(execution_id: str, status: str, output_data: Optional[Dict[str, Any]] = None, error_details: Optional[str] = None):
    """Atualiza o status final ou intermediário do workflow."""
    data = {"status": status, "updated_at": get_utc_now()}
    if output_data:
        data["output_data"] = output_data
    if error_details:
        data["error_details"] = error_details
    if status in ["SUCCESS", "FAILED"]:
        data["completed_at"] = get_utc_now()
    if status == "RUNNING":
        data["started_at"] = get_utc_now()

    try:
        supabase.table("workflow_executions").update(data).eq("id", execution_id).execute()
    except Exception as e:
        logger.error(f"Erro ao atualizar status do workflow: {e}")

async def run_step_with_retry(
    execution_id: str,
    step_name: str,
    worker_func: Optional[Callable] = None,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """
    Executor genérico de nós com retry, backoff e rastreabilidade.
    """
    attempt = 1
    last_error = None

    while attempt <= max_retries:
        try:
            logger.info(f"Executando step {step_name} (tentativa {attempt}/{max_retries})...")
            
            result = None
            if worker_func:
                if asyncio.iscoroutinefunction(worker_func):
                    result = await worker_func(**kwargs)
                else:
                    result = worker_func(**kwargs)
            
            # Registro de sucesso
            step_data = {
                "execution_id": execution_id,
                "step_name": step_name,
                "attempt": attempt,
                "status": "SUCCESS",
                "output_data": {"result": result} if result else None,
                "completed_at": get_utc_now()
            }
            supabase.table("workflow_step_executions").insert(step_data).execute()
            return result

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Erro no step {step_name} ({attempt}/{max_retries}): {last_error}")
            
            # Registro de falha
            step_data = {
                "execution_id": execution_id,
                "step_name": step_name,
                "attempt": attempt,
                "status": "FAILED",
                "error_details": last_error,
                "completed_at": get_utc_now()
            }
            supabase.table("workflow_step_executions").insert(step_data).execute()

            if attempt < max_retries:
                # Exponential backoff + jitter
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                await asyncio.sleep(sleep_time)
            
            attempt += 1

    raise Exception(f"Step {step_name} falhou após {max_retries} tentativas. Último erro: {last_error}")

def get_next_occurrence_of_hour(hour: int) -> datetime:
    """
    Calcula o próximo datetime (UTC) correspondente à hora informada (0-23) no fuso de Brasília.
    """
    from datetime import timedelta
    now_br = get_br_now()
    target_br = now_br.replace(hour=hour, minute=0, second=0, microsecond=0)
    
    if target_br <= now_br:
        target_br += timedelta(days=1)
        
    return target_br.astimezone(timezone.utc)
