import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    # Em produção, isso garantirá que vejamos o erro nos logs ou falha no boot
    raise ValueError("Variáveis de ambiente SUPABASE_URL ou SUPABASE_KEY não configuradas no Easypanel.")

# Singleton do Supabase Client
supabase: Client = create_client(supabase_url, supabase_key)
