import pandas as pd
import numpy as np
import zoneinfo

BR_TZ = zoneinfo.ZoneInfo("America/Sao_Paulo")

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

LS_FEATURES = [
    'ddd', 'Regiao',
    'n_tentativas_anteriores', 'horas_desde_primeiro_contato',
    'n_voicemail_reached_anteriores', 'n_dial_no_answer_anteriores',
    'n_inactivity_anteriores', 'N_invalid_destination_anteriores',
    'N_user_hangup_anteriores', 'N_user_declined_anteriores',
    'N_telephony_provider_permission_denied_anteriores', 'N_dial_busy_anteriores',
    'N_telephony_provider_unavailable_anteriores', 'N_agent_hangup_anteriores',
    'N_error_asr_anteriores', 'N_error_retell_anteriores',
    'N_dial_failed_anteriores', 'N_max_duration_reached_anteriores',
    'N_ivr_reached_anteriores',
    'ultima_disconnection_reason'
]

TP_FEATURES = [
    'ddd',
    'hora_do_contato',
    'dia_da_semana',
    'hora_ultimo_contato',
    'densidade_tentativas',
    'pressao_recente'
]

def get_ddd(phone):
    phone = str(phone).replace('+', '').replace('55', '')
    if not phone or len(phone) < 2: return "00"
    return phone[:2]

def transform_features(numero: str, history_data: list) -> pd.DataFrame:
    """
    Transforma o histórico do lead em features para os modelos XGBoost.
    """
    # 1. Preparar DataFrame
    if not history_data:
        df = pd.DataFrame(columns=['created_at', 'disconnection_reason'])
    else:
        df = pd.DataFrame(history_data)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert(BR_TZ)
        df = df.sort_values('created_at').reset_index(drop=True)

    # 2. Adicionar linha "dummy" para a predição atual (agora)
    now = pd.Timestamp.now(tz=BR_TZ)
    current_row = {
        'created_at': now,
        'disconnection_reason': 'ML_PREDICT_PENDING'
    }
    df = pd.concat([df, pd.DataFrame([current_row])], ignore_index=True)

    # 3. Features Geográficas
    df['ddd'] = get_ddd(numero)
    df['Regiao'] = df['ddd'].map(DDD_TO_STATE).fillna('Outros')

    # 4. Features Temporais
    df['hora_do_contato'] = df['created_at'].dt.hour
    df['dia_da_semana'] = df['created_at'].dt.dayofweek

    # 5. Métricas de Tentativas
    df['n_tentativas_anteriores'] = df.index
    first_contact = df['created_at'].iloc[0]
    df['horas_desde_primeiro_contato'] = (df['created_at'] - first_contact).dt.total_seconds() / 3600
    df['horas_desde_ultimo_contato'] = (df['created_at'] - df['created_at'].shift(1)).dt.total_seconds() / 3600
    df['hora_ultimo_contato'] = df['hora_do_contato'].shift(1).fillna(-1)

    # 6. Fadiga e Pressão
    df['densidade_tentativas'] = (df['n_tentativas_anteriores'] / (df['horas_desde_primeiro_contato'] + 1)).round(4)
    df['pressao_recente'] = (df['n_tentativas_anteriores'] / (df['horas_desde_ultimo_contato'].fillna(0) + 1)).round(4)

    # 7. Contagem de Motivos de Desconexão (Acumulado)
    reasons = [
        'voicemail_reached', 'dial_no_answer', 'inactivity', 'invalid_destination',
        'user_hangup', 'user_declined', 'telephony_provider_permission_denied',
        'dial_busy', 'telephony_provider_unavailable', 'agent_hangup', 'error_asr',
        'error_retell', 'dial_failed', 'max_duration_reached', 'ivr_reached'
    ]

    for reason in reasons:
        # Convenção de nomes no workflow.md
        if reason in ['voicemail_reached', 'dial_no_answer', 'inactivity']:
            col_name = f'n_{reason}_anteriores'
        else:
            col_name = f'N_{reason}_anteriores'
        
        is_reason = (df['disconnection_reason'] == reason).astype(int)
        df[col_name] = is_reason.cumsum().shift(1).fillna(0)

    # 8. Última disconnection reason
    df['ultima_disconnection_reason'] = df['disconnection_reason'].shift(1).fillna('primeiro_contato')

    # Retorna apenas a última linha (a do ML_PREDICT_PENDING)
    final_df = df.tail(1).copy()
    
    # Garantir que todas as colunas necessárias existam (fillna se necessário)
    for col in LS_FEATURES + TP_FEATURES:
        if col not in final_df.columns:
            final_df[col] = 0

    return final_df
