import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Singleton do Supabase Client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
