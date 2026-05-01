import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

def genera_dati_innovazione(folder_path, output_csv):
  # Preparazione della lista dei file
  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  if not files:
    print(f"Errore: Nessun file .tsv.gz trovato in {folder_path}, scaricare dati con data_import.py")
    return

  # Randomizzazione dei file, altrimenti sarebbero in ordine di nome
  random.shuffle(files)
  print(f"Inizio analisi su {len(files)} file...")

  # Inizializzazione variabili globali
  n_totale = 0           # Taglia cumulativa (n)
  visti_famiglie = set() # Vocabolario unico famiglie (F_n)
  visti_clan = set()     # Vocabolario unico clan
  next_checkpoint = 10  # Partiamo da un valore piccolo
  fattore_crescita = 1.1 # Incremento del 10% per ogni nuovo punto
  
  risultati = []

  # Ciclo sui file
  for f in tqdm(files, desc="Processando genomi"):
    try:
      # Leggiamo il file (Colonna 5: hmm_acc, Colonna 13: clan)
      # Usiamo comment='#' per saltare le righe di intestazione
      df = pd.read_csv(
        f, 
        sep='\t', 
        comment='#', 
        header=None, 
        usecols=[5, 13], 
        names=['family', 'clan'],
        engine='c',
        na_values=['No_clan','\\N', ''] 
      )

      # Usiamo zip per scorrere contemporaneamente famiglie e clan riga per riga
      for fam, cl in zip(df['family'], df['clan']):
        n_totale += 1
        
        # Aggiornamento Famiglie
        visti_famiglie.add(fam)
        
        # Aggiornamento Clan (se non è nullo/No_clan)
        if pd.notna(cl):
          visti_clan.add(cl)
        
        # Checkpoint logaritmico perfettamente sincronizzato
        if n_totale >= next_checkpoint:
          risultati.append({
            'n': n_totale,
            'Fn_family': len(visti_famiglie),
            'Fn_clan': len(visti_clan)
          })
          next_checkpoint = int(next_checkpoint * fattore_crescita) + 1

    except Exception as e:
      print(f"Errore nel file {f.name}: {e}")
      continue

  risultati.append({
  'n': n_totale,
  'Fn_family': len(visti_famiglie),
  'Fn_clan': len(visti_clan)
  })

  # Salvataggio risultati
  df_risultati = pd.DataFrame(risultati)
  df_risultati.to_csv(output_csv, index=False)
  print(f"\nAnalisi completata! Dati salvati in: {output_csv}")
  print(f"Taglia finale (n): {n_totale}")
  print(f"Vocabolario finale (F_n): {len(visti_famiglie)}")

if __name__ == "__main__":
  cartella_dati = "dati_proteomi" 
  nome_output = "risultati_heaps_innovazione.csv"
  
  genera_dati_innovazione(cartella_dati, nome_output)