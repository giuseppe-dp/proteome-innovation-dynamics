import pandas as pd
import asyncio
import httpx
import os
from pathlib import Path
from tqdm.asyncio import tqdm

def filter_gtdb_metadata(file_path, output_path):
  print(f"--- Inizio elaborazione di: {file_path} ---")
  
  # Selezioniamo solo le colonne che ci servono per risparmiare memoria
  cols_to_use = [
    'accession', 
    'gtdb_representative', 
    'checkm2_completeness', 
    'checkm2_contamination', 
    'ncbi_taxid'
  ]
  
  # Caricamento del file
  # Usiamo sep='\t' perché è un TSV
  try:
    df = pd.read_csv(file_path, sep='\t', usecols=cols_to_use, low_memory=False)
  except ValueError as e:
    print(f"Errore: Assicurati che i nomi delle colonne siano corretti\n{e}")
    return

  print(f"Righe totali nel dataset: {len(df)}")

  # Applicazione dei filtri
  # 1. Deve essere un genoma rappresentativo
  # 2. Completezza > 90%
  # 3. Contaminazione < 5%
  filtered_df = df[
    (df['gtdb_representative'] == 't') & 
    (df['checkm2_completeness'] > 90) & 
    (df['checkm2_contamination'] < 5)
  ].copy()

  # Rimuoviamo eventuali righe senza ncbi_taxid (fondamentali per Pfam)
  filtered_df = filtered_df.dropna(subset=['ncbi_taxid'])
  
  # Convertiamo ncbi_taxid in intero
  filtered_df['ncbi_taxid'] = filtered_df['ncbi_taxid'].astype(int)

  print(f"Righe dopo il filtraggio: {len(filtered_df)}")

  # Salvataggio della lista dei TaxID (ci servirà per scaricare i file Pfam)
  filtered_df[['accession', 'ncbi_taxid']].to_csv(output_path, index=False, sep='\t')
  
  print(f"--- Fatto! Lista salvata in: {output_path} ---")
  # return della lista con parametri taxid unici
  return filtered_df['ncbi_taxid'].unique().tolist()


async def download_file(client, taxid, save_dir, semaphore):
  # Il semaforo limita la concorrenza reale (es. 5 download alla volta)
  async with semaphore:
    url = f"https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/proteomes/{taxid}.tsv.gz"
    file_path = save_dir / f"{taxid}.tsv.gz"
    
    if file_path.exists():
      return "ESISTE"
  
    try:
      # Usiamo un timeout ragionevole
      response = await client.get(url, timeout=20.0)
      
      if response.status_code == 200:
        with open(file_path, "wb") as f:
          f.write(response.content)
        return "OK"
      elif response.status_code == 404:
        return "NON_TROVATO"
      else:
        return f"ERRORE_{response.status_code}"
    except Exception as e:
      return f"ECCEZIONE: {type(e).__name__}"
    

async def download_proteomi(taxid_list):
  save_dir = Path("dati_proteomi")
  save_dir.mkdir(exist_ok=True)
  
  # Troppe connessioni simultanee causano il blocco, ne faccio 5
  sem = asyncio.Semaphore(5) 
  
  # Configurazione client: un po' di "respiro" tra le connessioni
  limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
  
  async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
    print(f"--- AVVIO Download SU {len(taxid_list)} TAXID ---")
    
    tasks = [download_file(client, tid, save_dir, sem) for tid in taxid_list]
    
    # tqdm ci mostra il progresso in tempo reale
    results = await tqdm.gather(*tasks)
    
    # Analisi veloce dei risultati
    print("\n--- RISULTATI Download ---")
    print(f"Scaricati con successo: {results.count('OK')}")
    print(f"Non presenti su Pfam: {results.count('NON_TROVATO')}")
    print(f"Errori/Eccezioni: {len([r for r in results if 'ERRORE' in r or 'ECCEZIONE' in r])}")


def main():
  path_metadata = 'bac120_metadata.tsv.gz'
  taxid_list = filter_gtdb_metadata(path_metadata, 'lista_taxid_batteri_alta_qualita.tsv')
  asyncio.run(download_proteomi(taxid_list))

if __name__ == "__main__":
  main()