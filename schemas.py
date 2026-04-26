from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class PredictWebhookInput(BaseModel):
    """Payload de entrada para o /webhook/predict."""
    numero: str = Field(..., description="Número do lead no formato +55DDXXXXXXXXX")
    agent_id: str = Field(..., description="ID do agente Retell associado")
    nome: str = Field(..., description="Nome completo do lead")
    email: str = Field(..., description="E-mail do lead")
    Prompt_id: str = Field(..., description="ID do prompt a ser usado no pre_call_processing")
    contexto: Optional[str] = Field(None, description="Contexto adicional do lead")
    empresa: Optional[str] = Field(None, description="Nome da empresa do lead")
    segmento: Optional[str] = Field(None, description="Segmento de negócio do lead")

class PreCallOutput(BaseModel):
    """Payload de saída para o workflow pre_call_processing."""
    workflow_name: str = "pre_call_processing"
    execution_id: str
    numero: str
    nome: str
    email: str
    agent_id: str
    Prompt_id: str
    contexto: Optional[str] = None
    empresa: Optional[str] = None
    segmento: Optional[str] = None
    quando_ligar: Optional[str] = None  # ISO 8601 com timezone

class MLFeaturesLS(BaseModel):
    """Vetor de features para o modelo Lead Scoring."""
    ddd: str
    Regiao: str
    n_tentativas_anteriores: int
    horas_desde_primeiro_contato: float
    n_voicemail_reached_anteriores: int
    n_dial_no_answer_anteriores: int
    n_inactivity_anteriores: int
    N_invalid_destination_anteriores: int
    N_user_hangup_anteriores: int
    N_user_declined_anteriores: int
    N_telephony_provider_permission_denied_anteriores: int
    N_dial_busy_anteriores: int
    N_telephony_provider_unavailable_anteriores: int
    N_agent_hangup_anteriores: int
    N_error_asr_anteriores: int
    N_error_retell_anteriores: int
    N_dial_failed_anteriores: int
    N_max_duration_reached_anteriores: int
    N_ivr_reached_anteriores: int
    ultima_disconnection_reason: str

class MLFeaturesTP(BaseModel):
    """Vetor de features para o modelo Timing Predict."""
    ddd: str
    hora_contato: int
    dia_semana: int
    hora_ultimo_contato: float
    densidade_tentativas: float
    pressao_recente: float
