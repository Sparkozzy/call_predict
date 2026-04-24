import os
import random
import httpx
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from supabase import create_client, Client
from schemas import PredictWebhookInput, PreCallOutput
from ml_logic import (
    BRT, transform_to_ls_features, transform_to_tp_features,
    run_ls_inference, run_tp_simulation
)

load_dotenv()

logger = logging.getLogger(__name__)

# --- CLIENTES ---
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# --- AUXILIARES DE RASTREABILIDADE ---

async def create_step(execution_id: str, step_name: str, input_data: Dict[str, Any] = None):
    res = supabase.table("workflow_step_executions").insert({
        "execution_id": execution_id,
        "step_name": step_name,
        "status": "RUNNING",
        "input_data": input_data,
        "started_at": datetime.now(ZoneInfo("UTC")).isoformat()
    }).execute()
    return res.data[0]["id"]

async def finish_step(step_id: str, status: str, output_data: Dict[str, Any] = None, error: str = None):
    supabase.table("workflow_step_executions").update({
        "status": status,
        "output_data": output_data,
        "error_details": error,
        "completed_at": datetime.now(ZoneInfo("UTC")).isoformat()
    }).eq("id", step_id).execute()

# --- ORQUESTRADOR PRINCIPAL ---

async def process_call_predict(ctx, data: PredictWebhookInput, execution_id: str):
    """
    Executa o workflow call_predict completo (10 nós) com rastreabilidade total.
    """
    try:
        # Atualizar workflow para RUNNING
        supabase.table("workflow_executions").update({
            "status": "RUNNING",
            "started_at": datetime.now(ZoneInfo("UTC")).isoformat()
        }).eq("id", execution_id).execute()

        # [Nó 2] Exploitation Decision
        rate = float(os.getenv("EXPLORATION_RATE", "0.05"))
        step_exp = await create_step(execution_id, "call_predict_exploitation", {"exploration_rate": rate})
        is_exploration = random.random() < rate
        await finish_step(step_exp, "SUCCESS", {"is_exploration": is_exploration})

        if is_exploration:
            await handle_exploration_path(ctx, data, execution_id)
            return

        # [Nó 3] Get Rows
        step_get = await create_step(execution_id, "call_predict_get_rows", {"numero": data.numero, "limit": 150})
        res = supabase.table("Retell_calls_Mindflow")\
            .select("to_number, created_at, disconnection_reason")\
            .eq("to_number", data.numero)\
            .order("created_at", desc=True)\
            .limit(150)\
            .execute()
        
        rows = list(reversed(res.data))
        tem_historico = len(rows) > 0
        await finish_step(step_get, "SUCCESS", {"rows_count": len(rows), "tem_historico": tem_historico})

        ls_prob = None
        tp_hora_escolhida = None
        tp_prob_pico = None
        quando_ligar_iso = None

        if tem_historico:
            # [Nó 4] Transform LS
            step_t_ls = await create_step(execution_id, "call_predict_data_transform_ls", {"rows_sample": rows[:3]})
            features_ls = transform_to_ls_features(data.numero, rows)
            await finish_step(step_t_ls, "SUCCESS", {"features": features_ls.dict()})

            # [Nó 5] Run LS
            step_r_ls = await create_step(execution_id, "call_predict_run_ls", {"features": features_ls.dict()})
            ls_prob = run_ls_inference(ctx["model_ls"], features_ls)
            threshold = float(os.getenv("LS_THRESHOLD", "0.0045"))
            ls_decisao = "LIGAR" if ls_prob >= threshold else "DESCARTAR"
            await finish_step(step_r_ls, "SUCCESS", {"probabilidade": ls_prob, "decisao": ls_decisao, "threshold": threshold})

            # Registrar inferência
            model_exec_res = supabase.table("model_executions").insert({
                "model_id": "lead_scoring_v1",
                "model_name": "XGBoost Lead Scoring",
                "model_version": "1.0.0",
                "to_number": data.numero,
                "agent_id": data.agent_id,
                "ls_probabilidade": ls_prob,
                "ls_decisao": ls_decisao,
                "is_exploration": False
            }).execute()
            model_exec_id = model_exec_res.data[0]["id"]

            # [Nó 6] LS Threshold
            step_gate = await create_step(execution_id, "call_predict_ls_threshold", {"ls_prob": ls_prob, "threshold": threshold})
            if ls_decisao == "DESCARTAR":
                await finish_step(step_gate, "SUCCESS", {"passou": False, "motivo": "Abaixo do threshold"})
                await finalize_workflow(execution_id, "SUCCESS", {"decisao": "DESCARTAR", "motivo": "ls_threshold"})
                return
            await finish_step(step_gate, "SUCCESS", {"passou": True})
        else:
            # Lead sem histórico
            model_exec_res = supabase.table("model_executions").insert({
                "model_id": "timing_predict_v1",
                "model_name": "XGBoost Timing Predict (No History)",
                "model_version": "1.0.0",
                "to_number": data.numero,
                "agent_id": data.agent_id,
                "is_exploration": False
            }).execute()
            model_exec_id = model_exec_res.data[0]["id"]

        # [Nó 7] Transform TP
        now_br = datetime.now(BRT)
        step_t_tp = await create_step(execution_id, "call_predict_data_transform_tp", {"now_br": now_br.isoformat(), "rows_count": len(rows)})
        features_tp = transform_to_tp_features(data.numero, rows, now_br)
        await finish_step(step_t_tp, "SUCCESS", {"features": features_tp.dict()})

        # [Nó 8] Run TP
        step_r_tp = await create_step(execution_id, "call_predict_run_tp", {"features": features_tp.dict()})
        melhor = run_tp_simulation(ctx["model_tp"], features_tp)
        
        agendamento = now_br + timedelta(hours=melhor["offset"])
        agendamento = agendamento.replace(minute=0, second=0, microsecond=0)
        quando_ligar_iso = agendamento.isoformat()
        
        await finish_step(step_r_tp, "SUCCESS", {
            "melhor_hora": melhor["hora"],
            "quando_ligar": quando_ligar_iso,
            "prob_pico": melhor["probabilidade"]
        })

        # Atualizar model_executions
        supabase.table("model_executions").update({
            "tp_horario_escolhido": melhor["hora"],
            "tp_probabilidade_pico": melhor["probabilidade"]
        }).eq("id", model_exec_id).execute()

        # [Nó 9 & 10] Send Payload
        await send_to_mindflow(execution_id, data, quando_ligar_iso)

    except Exception as e:
        logger.exception(f"Erro no processamento do lead {data.numero}")
        await finalize_workflow(execution_id, "FAILED", error=str(e))

async def handle_exploration_path(ctx, data: PredictWebhookInput, execution_id: str):
    """Grupo de controle (Exploration)."""
    step_exp_path = await create_step(execution_id, "call_predict_exploration_logic", {"numero": data.numero})
    
    supabase.table("model_executions").insert({
        "model_id": "exploration_control",
        "to_number": data.numero,
        "agent_id": data.agent_id,
        "is_exploration": True
    }).execute()

    now_br = datetime.now(BRT)
    TP_BLOCKED_START = int(os.getenv("TP_BLOCKED_START", "23"))
    TP_BLOCKED_END = int(os.getenv("TP_BLOCKED_END", "6"))
    
    hora_atual = now_br.hour
    agendamento = now_br + timedelta(hours=1) # Default próxima hora
    
    if hora_atual >= TP_BLOCKED_START or hora_atual <= TP_BLOCKED_END:
        agendamento = now_br.replace(hour=TP_BLOCKED_END + 1, minute=0, second=0, microsecond=0)
        if agendamento <= now_br:
            agendamento += timedelta(days=1)
    
    quando_ligar_iso = agendamento.isoformat()
    await finish_step(step_exp_path, "SUCCESS", {"quando_ligar": quando_ligar_iso})
    await send_to_mindflow(execution_id, data, quando_ligar_iso)

async def send_to_mindflow(execution_id: str, input_data: PredictWebhookInput, quando_ligar: Optional[str]):
    """Nós 9 e 10: Cria e envia payload."""
    payload = {
        "workflow_name": "pre_call_processing",
        "execution_id": execution_id,
        "numero": input_data.numero,
        "nome": input_data.nome,
        "email": input_data.email,
        "agent_id": input_data.agent_id,
        "Prompt_id": input_data.Prompt_id, # Garantido aqui
        "quando_ligar": quando_ligar
    }
    
    step_send = await create_step(execution_id, "call_predict_send", {"target_url": os.getenv("MINDFLOW_WEBHOOK_URL"), "payload": payload})
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                os.getenv("MINDFLOW_WEBHOOK_URL"), 
                json=payload,
                headers={"X-API-Key": os.getenv("WEBHOOK_API_KEY"), "Content-Type": "application/json"},
                timeout=30.0
            )
            response.raise_for_status()
            await finish_step(step_send, "SUCCESS", {"status_code": response.status_code, "response": response.text})
            await finalize_workflow(execution_id, "SUCCESS", {"decisao": "LIGAR", "quando_ligar": quando_ligar})
        except Exception as e:
            await finish_step(step_send, "FAILED", error=str(e))
            raise e

async def finalize_workflow(execution_id: str, status: str, output: Dict[str, Any] = None, error: str = None):
    supabase.table("workflow_executions").update({
        "status": status,
        "output_data": output,
        "error_details": error,
        "completed_at": datetime.now(ZoneInfo("UTC")).isoformat()
    }).eq("id", execution_id).execute()
