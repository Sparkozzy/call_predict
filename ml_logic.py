import os
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
from schemas import MLFeaturesLS, MLFeaturesTP

# --- CONFIGURAÇÕES E METADADOS ---

BRT = ZoneInfo("America/Sao_Paulo")

DDD_TO_STATE = {
    '11': 'SP', '12': 'SP', '13': 'SP', '14': 'SP', '15': 'SP', '16': 'SP', '17': 'SP', '18': 'SP', '19': 'SP',
    '21': 'RJ', '22': 'RJ', '24': 'RJ', '27': 'ES', '28': 'ES', '31': 'MG', '32': 'MG', '33': 'MG', '34': 'MG',
    '35': 'MG', '37': 'MG', '38': 'MG', '41': 'PR', '42': 'PR', '43': 'PR', '44': 'PR', '45': 'PR', '46': 'PR',
    '47': 'SC', '48': 'SC', '49': 'SC', '51': 'RS', '53': 'RS', '54': 'RS', '55': 'RS', '61': 'DF', '62': 'GO',
    '63': 'TO', '64': 'GO', '65': 'MT', '66': 'MT', '67': 'MS', '68': 'AC', '69': 'RO', '71': 'BA', '73': 'BA',
    '74': 'BA', '75': 'BA', '77': 'BA', '79': 'SE', '81': 'PE', '82': 'AL', '83': 'PB', '84': 'RN', '85': 'CE',
    '86': 'PI', '87': 'PE', '88': 'CE', '89': 'PI', '91': 'PA', '92': 'AM', '93': 'PA', '94': 'PA', '95': 'RR',
    '96': 'AP', '97': 'AM', '98': 'MA', '99': 'MA'
}

# Categorias exatas do treinamento (para garantir mapping correto no XGBoost)
CAT_DDD = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '24', '27', '28', '29', '31', '32', '33', '34', '35', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48', '49', '51', '53', '54', '59', '61', '62', '63', '64', '65', '66', '67', '68', '69', '71', '73', '74', '75', '77', '79', '81', '82', '83', '84', '85', '86', '87', '88', '89', '91', '92', '94', '95', '97', '98', '99', 'na']
CAT_REGIAO = ['AC', 'AL', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP']
CAT_DISCONNECT = ['agent_hangup', 'dial_busy', 'dial_failed', 'dial_no_answer', 'error_asr', 'error_retell', 'inactivity', 'invalid_destination', 'ivr_reached', 'max_duration_reached', 'primeiro_contato', 'telephony_provider_permission_denied', 'telephony_provider_unavailable', 'user_declined', 'user_hangup', 'voicemail_reached']
CAT_DIA_SEMANA = [0, 1, 2, 3, 4, 5, 6]

REASONS_LIST = [r for r in CAT_DISCONNECT if r != "primeiro_contato"]

# --- LÓGICA DE ETL ---

def transform_to_ls_features(numero: str, rows: List[Dict[str, Any]]) -> MLFeaturesLS:
    """
    Transforma histórico do Supabase em features para o Lead Scoring.
    Assume que 'rows' já foi revertida para ordem cronológica (ASC).
    """
    now_br = datetime.now(BRT)
    
    # Mapear 'no-answer' -> 'dial_no_answer'
    for r in rows:
        if r.get("disconnection_reason") == "no-answer":
            r["disconnection_reason"] = "dial_no_answer"

    ddd = numero[3:5]
    regiao = DDD_TO_STATE.get(ddd, "na")
    
    if not rows:
        # Pela decisão Q5, o nó LS nem deveria rodar para leads sem histórico,
        # mas mantemos o default por segurança se for chamado.
        return MLFeaturesLS(
            ddd=ddd, Regiao=regiao, n_tentativas_anteriores=0,
            horas_desde_primeiro_contato=0.0, ultima_disconnection_reason="primeiro_contato",
            **{f"n_{r}_anteriores": 0 for r in REASONS_LIST},
            **{f"N_{r}_anteriores": 0 for r in REASONS_LIST} # Suporta os dois prefixos n_ e N_ do workflow.md
        )

    n_tentativas = len(rows)
    # Parse da data do banco (UTC) para BRT
    primeiro_at = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00")).astimezone(BRT)
    horas_desde_primeiro = (now_br - primeiro_at).total_seconds() / 3600
    
    ultima_reason = rows[-1].get("disconnection_reason") or "primeiro_contato"
    if ultima_reason not in CAT_DISCONNECT:
        ultima_reason = "dial_no_answer" # Fallback conservativo

    # Contagens de razões
    counts = {}
    for r in REASONS_LIST:
        cnt = sum(1 for row in rows if row.get("disconnection_reason") == r)
        counts[f"n_{r}_anteriores"] = cnt
        counts[f"N_{r}_anteriores"] = cnt # Garantir consistência com o que o modelo espera

    return MLFeaturesLS(
        ddd=ddd,
        Regiao=regiao,
        n_tentativas_anteriores=n_tentativas,
        horas_desde_primeiro_contato=round(horas_desde_primeiro, 4),
        ultima_disconnection_reason=ultima_reason,
        **counts
    )

def transform_to_tp_features(numero: str, rows: List[Dict[str, Any]], now_br: datetime) -> MLFeaturesTP:
    """
    Transforma histórico em features para o Timing Predict.
    """
    ddd = numero[3:5]
    hora_contato = now_br.hour
    dia_semana = now_br.weekday()
    
    if not rows:
        n_tentativas = 0
        horas_desde_primeiro = 0.0
        hora_ultimo = -1.0
        horas_desde_ultimo = 0.0
    else:
        n_tentativas = len(rows)
        primeiro_at = datetime.fromisoformat(rows[0]["created_at"].replace("Z", "+00:00")).astimezone(BRT)
        ultimo_at = datetime.fromisoformat(rows[-1]["created_at"].replace("Z", "+00:00")).astimezone(BRT)
        
        horas_desde_primeiro = (now_br - primeiro_at).total_seconds() / 3600
        horas_desde_ultimo = (now_br - ultimo_at).total_seconds() / 3600
        hora_ultimo = float(ultimo_at.hour)

    densidade = n_tentativas / (horas_desde_primeiro + 1)
    pressao = n_tentativas / (horas_desde_ultimo + 1)

    return MLFeaturesTP(
        ddd=ddd,
        hora_contato=hora_contato,
        dia_semana=dia_semana,
        hora_ultimo_contato=hora_ultimo,
        densidade_tentativas=round(densidade, 4),
        pressao_recente=round(pressao, 4)
    )

# --- LÓGICA DE INFERÊNCIA ---

def run_ls_inference(model: Any, features: MLFeaturesLS) -> float:
    """Roda inferência do Lead Scoring com tratamento rigoroso de categorias."""
    data = features.dict()
    # Remover campos duplicados n_ vs N_ se necessário para o DataFrame
    # O modelo espera 20 features exatas conforme inspeção
    feature_order = [
        'ddd', 'Regiao', 'n_tentativas_anteriores', 'horas_desde_primeiro_contato',
        'n_voicemail_reached_anteriores', 'n_dial_no_answer_anteriores', 'n_inactivity_anteriores',
        'N_invalid_destination_anteriores', 'N_user_hangup_anteriores', 'N_user_declined_anteriores',
        'N_telephony_provider_permission_denied_anteriores', 'N_dial_busy_anteriores',
        'N_telephony_provider_unavailable_anteriores', 'N_agent_hangup_anteriores',
        'N_error_asr_anteriores', 'N_error_retell_anteriores', 'N_dial_failed_anteriores',
        'N_max_duration_reached_anteriores', 'N_ivr_reached_anteriores', 'ultima_disconnection_reason'
    ]
    
    df = pd.DataFrame([data])[feature_order]
    
    # Tratamento Categórico Crítico
    df['ddd'] = pd.Categorical(df['ddd'], categories=CAT_DDD)
    df['Regiao'] = pd.Categorical(df['Regiao'], categories=CAT_REGIAO)
    df['ultima_disconnection_reason'] = pd.Categorical(df['ultima_disconnection_reason'], categories=CAT_DISCONNECT)
    
    prob = model.predict_proba(df)[0][1]
    return float(prob)

def run_tp_simulation(model: Any, base_features: MLFeaturesTP) -> Dict[str, Any]:
    """Simula 24h para encontrar o melhor horário (Timing Predict)."""
    TP_FORECAST_HOURS = int(os.getenv("TP_FORECAST_HOURS", "24"))
    TP_BLOCKED_START = int(os.getenv("TP_BLOCKED_START", "23"))
    TP_BLOCKED_END = int(os.getenv("TP_BLOCKED_END", "6"))
    
    hora_atual = base_features.hora_contato
    dia_atual = base_features.dia_semana
    
    resultados = []
    
    for i in range(1, TP_FORECAST_HOURS + 1):
        prox_hora = (hora_atual + i) % 24
        prox_dia = (dia_atual + ((hora_atual + i) // 24)) % 7
        
        # Pular bloqueados
        if prox_hora >= TP_BLOCKED_START or prox_hora <= TP_BLOCKED_END:
            continue
            
        sim_data = base_features.dict()
        sim_data['hora_contato'] = prox_hora
        sim_data['dia_semana'] = prox_dia
        
        df = pd.DataFrame([sim_data])
        
        # Tratamento Categórico
        df['ddd'] = pd.Categorical(df['ddd'], categories=CAT_DDD)
        df['dia_semana'] = pd.Categorical(df['dia_semana'].astype(int), categories=CAT_DIA_SEMANA)
        
        prob = model.predict_proba(df)[0][1]
        
        resultados.append({
            "hora": prox_hora,
            "dia": prox_dia,
            "probabilidade": float(prob),
            "offset": i
        })
    
    if not resultados:
        # Fallback se algo der errado com a janela (improvável)
        return {"offset": 1}
        
    melhor = max(resultados, key=lambda x: x["probabilidade"])
    return melhor
