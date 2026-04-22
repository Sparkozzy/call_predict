"""
schemas.py — Modelos Pydantic do projeto call_predict.

Toda entrada e saída de endpoints e tasks deve ter um schema aqui.
"""
from typing import Optional
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Payload de entrada — POST /webhook/predict
# ---------------------------------------------------------------------------

class CallPredictPayload(BaseModel):
    """
    Payload mínimo exigido pelo endpoint /webhook/predict.

    Campos:
        numero    — Número de telefone do lead (formato E.164, ex: +5511999999999)
        agent_id  — Identificador do agente Retell que fará a ligação
    """
    numero: str
    agent_id: str

    @field_validator("numero")
    @classmethod
    def numero_deve_ter_prefixo_internacional(cls, v: str) -> str:
        if not v.startswith("+"):
            raise ValueError("O campo 'numero' deve começar com '+' (formato E.164).")
        return v


# ---------------------------------------------------------------------------
# Schema de rastreabilidade — tabela model_executions (Supabase)
# Usado como referência de schema. A tabela já existe no banco.
# ---------------------------------------------------------------------------

class ModelExecutionSchema(BaseModel):
    """
    Representa uma linha da tabela `model_executions` no Supabase.

    Campos marcados como Optional são preenchidos apenas se o step
    correspondente foi executado (ex: ls_* só existem se is_exploration=False).
    """
    # Chave primária (gerada pelo banco)
    # id: uuid — gerado automaticamente via gen_random_uuid()

    # Identificação do modelo
    model_id: str
    model_name: str
    model_version: str

    # Lead avaliado
    to_number: str
    agent_id: str

    # Resultado Lead Scoring (nullable se exploration ou descartado antes do TP)
    ls_probabilidade: Optional[float] = None
    ls_decisao: Optional[str] = None          # 'LIGAR' | 'DESCARTAR'

    # Resultado Timing Predict (nullable se LS retornou DESCARTAR ou exploration)
    tp_horario_escolhido: Optional[int] = None    # 0 a 23
    tp_probabilidade_pico: Optional[float] = None

    # Flag de grupo de controle
    is_exploration: bool


# ---------------------------------------------------------------------------
# Resposta do endpoint 202 Accepted
# ---------------------------------------------------------------------------

class AcceptedResponse(BaseModel):
    """Resposta padrão para endpoints que enfileiram tarefas assíncronas."""
    status: str = "accepted"
    message: str
    execution_id: str
