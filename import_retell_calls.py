import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("As credenciais do Supabase não foram encontradas no .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_all_calls():
    print("Iniciando extração da tabela Retell_calls_Mindflow...")
    all_data = []
    import time
    limit = 500
    offset = 0
    max_retries = 3
    
    while True:
        success = False
        for attempt in range(max_retries):
            try:
                response = supabase.table("Retell_calls_Mindflow").select("*").range(offset, offset + limit - 1).execute()
                data = response.data
                success = True
                break
            except Exception as e:
                print(f"Erro na extração (tentativa {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
                
        if not success:
            print("Falha ao extrair dados após várias tentativas. Abortando.")
            break
            
        if not data:
            break
            
        all_data.extend(data)
        print(f"Extraídos {len(data)} registros (Total: {len(all_data)})...")
        
        if len(data) < limit:
            break
            
        offset += limit
        
    df = pd.DataFrame(all_data)
    
    # Salva o resultado
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "Retell_calls_Mindflow.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nExtração concluída com sucesso! {len(df)} registros salvos em '{output_path}'.")
    return df

if __name__ == "__main__":
    df_calls = extract_all_calls()
    print("\nVisualização das primeiras linhas:")
    print(df_calls.head())
