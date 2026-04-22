import logging
import random
from typing import Any, Dict
import os
import httpx
import pandas as pd
import numpy as np
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
    
    numero = payload_dict.get("numero")
    agent_id = payload_dict.get("agent_id")

    logger.info(f"Iniciando process_call_predict ({execution_id}) para o número {numero}")

    # 1. Início de Execução: Atualizar status para RUNNING
    await update_workflow_status(execution_id, status="RUNNING")

    try:
        # Step: Sorteio Exploration (Grupo de Controle 5%)
        async def do_exploration_sorteio():
            is_exploration = random.random() < 0.05
            return {"is_exploration": is_exploration}

        sorteio_result = await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_sorteio_exploration",
            worker_func=do_exploration_sorteio
        )
        is_exploration = sorteio_result["is_exploration"]

        if is_exploration:
            logger.info(f"Execution {execution_id} sorteada para EXPLORATION. Pulando ML.")
            
            # Step: Persistir exploração
            async def persist_exploration():
                from utils import get_utc_now
                data = {
                    "model_name": "cascade_ls_tp",
                    "to_number": numero,
                    "agent_id": agent_id,
                    "is_exploration": True,
                    "created_at": get_utc_now()
                }
                supabase.table("model_executions").insert(data).execute()
                return "Exploração persistida"

            await run_step_with_retry(
                execution_id=execution_id,
                step_name="call_predict_persist_model",
                worker_func=persist_exploration
            )

            # Step: Disparo da Ligação (Exploração - Horário Padrão 10h)
            async def trigger_exploration_call():
                from utils import get_next_occurrence_of_hour
                defer_until = get_next_occurrence_of_hour(10) # 10h da manhã
                await ctx["redis"].enqueue_job(
                    "trigger_retell_call",
                    {"numero": numero, "agent_id": agent_id, "execution_id": execution_id, "is_exploration": True},
                    _defer_until=defer_until
                )
                return f"Ligação de exploração agendada para {defer_until.isoformat()}"

            await run_step_with_retry(
                execution_id=execution_id,
                step_name="call_predict_trigger_call",
                worker_func=trigger_exploration_call
            )

            await update_workflow_status(
                execution_id, 
                status="SUCCESS", 
                output_data={"is_exploration": True, "note": "Lead sorteado para grupo de controle. Ligação agendada para 10h."}
            )
            return {"status": "success", "execution_id": execution_id, "is_exploration": True}

        # Step: ETL de Features
        async def etl_features():
            # 1. Buscar histórico no Supabase
            # Trazemos os últimos 100 registros para o número do lead
            response = supabase.table("Retell_calls_Mindflow")\
                .select("created_at", "disconnection_reason")\
                .eq("Numero", numero)\
                .order("created_at", ascending=False)\
                .limit(100)\
                .execute()
            
            history_data = response.data if response.data else []
            
            # 2. Transformar em DataFrame de features
            from ml_logic import transform_features, LS_FEATURES, TP_FEATURES
            df_features = transform_features(numero, history_data)
            
            return {
                "features_ls": df_features[LS_FEATURES].to_dict(orient="records")[0],
                "features_tp": df_features[TP_FEATURES].to_dict(orient="records")[0]
            }

        etl_result = await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_etl_features",
            worker_func=etl_features
        )
        features_ls = etl_result["features_ls"]
        features_tp = etl_result["features_tp"]

        # Step: Lead Scoring
        async def lead_scoring():
            model_ls = ctx.get("model_ls")
            if not model_ls:
                raise Exception("Modelo Lead Scoring não carregado.")
            
            # Preparar DMatrix (XGBoost requer formato específico)
            import xgboost as xgb
            # Converter features categóricas para o formato que o modelo espera (se necessário)
            # Como o modelo LS usa 'Regiao' e 'ultima_disconnection_reason', precisamos garantir que sejam strings ou categóricos
            # O XGBoost 2.0+ suporta categorias se o DataFrame tiver o tipo category
            X = pd.DataFrame([features_ls])
            for col in ['Regiao', 'ultima_disconnection_reason']:
                X[col] = X[col].astype('category')
            
            # Predição de probabilidade
            # dmat = xgb.DMatrix(X, enable_categorical=True) # Se o modelo foi salvo como Booster
            # Se foi salvo via sklearn API:
            ls_prob = float(model_ls.predict_proba(X)[:, 1][0])
            ls_decisao = "LIGAR" if ls_prob > 0.5 else "DESCARTAR"
            
            return {"ls_probabilidade": ls_prob, "ls_decisao": ls_decisao}

        ls_result = await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_lead_scoring",
            worker_func=lead_scoring
        )
        ls_prob = ls_result["ls_probabilidade"]
        ls_decisao = ls_result["ls_decisao"]

        if ls_decisao == "DESCARTAR":
            logger.info(f"Execution {execution_id} DESCARTADA pelo Lead Scoring (prob: {ls_prob:.4f})")
            await update_workflow_status(
                execution_id, 
                status="SUCCESS", 
                output_data={
                    "ls_probabilidade": ls_prob,
                    "ls_decisao": ls_decisao,
                    "note": "Lead descartado pelo modelo de scoring."
                }
            )
            return {"status": "success", "execution_id": execution_id, "ls_decisao": ls_decisao}

        # Step: Timing Predict
        async def timing_predict():
            model_tp = ctx.get("model_tp")
            if not model_tp:
                raise Exception("Modelo Timing Predict não carregado.")
            
            X = pd.DataFrame([features_tp])
            # Predição (assume que o modelo retorna o índice da hora 0-23 com maior prob)
            tp_probs = model_tp.predict_proba(X)[0]
            tp_horario_escolhido = int(np.argmax(tp_probs))
            tp_probabilidade_pico = float(np.max(tp_probs))
            
            return {
                "tp_horario_escolhido": tp_horario_escolhido,
                "tp_probabilidade_pico": tp_probabilidade_pico
            }

        tp_result = await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_timing_predict",
            worker_func=timing_predict
        )
        tp_horario = tp_result["tp_horario_escolhido"]
        tp_prob = tp_result["tp_probabilidade_pico"]

        # Step: Persistência de resultados
        async def persist_model():
            data = {
                "model_name": "cascade_ls_tp",
                "to_number": numero,
                "agent_id": agent_id,
                "ls_probabilidade": ls_prob,
                "ls_decisao": ls_decisao,
                "tp_horario_escolhido": tp_horario,
                "tp_probabilidade_pico": tp_prob,
                "is_exploration": False,
                "created_at": get_utc_now()
            }
            supabase.table("model_executions").insert(data).execute()
            return "Resultados persistidos"

        from utils import get_utc_now
        await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_persist_model",
            worker_func=persist_model
        )

        # Step: Disparo da Ligação
        async def trigger_call():
            from utils import get_next_occurrence_of_hour
            defer_until = get_next_occurrence_of_hour(tp_horario)
            
            # Enfileira a tarefa de disparo real (Retell) no futuro
            await ctx["redis"].enqueue_job(
                "trigger_retell_call",
                {"numero": numero, "agent_id": agent_id, "execution_id": execution_id},
                _defer_until=defer_until
            )
            return f"Ligação agendada para {defer_until.isoformat()}"

        await run_step_with_retry(
            execution_id=execution_id,
            step_name="call_predict_trigger_call",
            worker_func=trigger_call
        )

        # Finalização: SUCCESS
        await update_workflow_status(
            execution_id, 
            status="SUCCESS", 
            output_data={
                "ls_probabilidade": ls_prob,
                "ls_decisao": ls_decisao,
                "tp_horario_escolhido": tp_horario,
                "tp_probabilidade_pico": tp_prob,
                "scheduled_at": "deferred"
            }
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

async def trigger_retell_call(ctx: Dict[Any, Any], payload: Dict[str, Any]):
    """
    Worker task: Efetivamente dispara a ligação via Retell AI.
    """
    numero = payload.get("numero")
    agent_id = payload.get("agent_id")
    execution_id = payload.get("execution_id")
    
    logger.info(f"Disparando ligação Retell para {numero} (Execution: {execution_id})")
    
    # TODO: Implementar chamada real para Retell API
    # node 'call_predict_trigger_call' enfileira esta tarefa.
    
    return {"status": "triggered", "numero": numero}
