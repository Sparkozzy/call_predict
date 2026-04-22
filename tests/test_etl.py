import pytest
import pandas as pd
from ml_logic import transform_features, LS_FEATURES, TP_FEATURES

# Mock data based on Retell_calls_Mindflow table structure
MOCK_HISTORY = [
    {
        "created_at": "2024-04-20T10:00:00+00:00",
        "Numero": "+5511999999999",
        "disconnection_reason": "dial_no_answer"
    },
    {
        "created_at": "2024-04-20T12:00:00+00:00",
        "Numero": "+5511999999999",
        "disconnection_reason": "voicemail_reached"
    }
]

def test_transform_features_metrics():
    numero = "+5511999999999"
    df = transform_features(numero, MOCK_HISTORY)
    
    assert len(df) == 1
    assert df['n_tentativas_anteriores'].iloc[0] == 2
    assert df['ddd'].iloc[0] == "11"
    assert df['Regiao'].iloc[0] == "SP"
    assert df['n_dial_no_answer_anteriores'].iloc[0] == 1
    assert df['n_voicemail_reached_anteriores'].iloc[0] == 1
    assert df['ultima_disconnection_reason'].iloc[0] == "voicemail_reached"

def test_transform_features_empty_history():
    numero = "+5511988888888"
    df = transform_features(numero, [])
    
    assert len(df) == 1
    assert df['n_tentativas_anteriores'].iloc[0] == 0
    assert df['ultima_disconnection_reason'].iloc[0] == "primeiro_contato"
