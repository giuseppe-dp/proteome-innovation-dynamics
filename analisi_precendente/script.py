import pandas as pd
import numpy as np
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


def analizza_prognosi_innovazione(csv_file):
  # Caricamento dati
  df = pd.read_csv(csv_file)
  
  # Calcolo del Tasso di Innovazione Locale (Domanda 2)
  # Calcoliamo la derivata discreta dF/dn
  df['dn'] = df['n'].diff()
  df['df_fam'] = df['Fn_family'].diff()
  
  # Tasso di innovazione: quante nuove famiglie per ogni dominio aggiunto
  df['innovation_rate'] = df['df_fam'] / df['dn']
  
  # Rimuoviamo i primi righi con NaN e valori nulli per il logaritmo
  df_prognosi = df.dropna()
  df_prognosi = df_prognosi[df_prognosi['innovation_rate'] > 0]

  return df_prognosi



def genera_modello_A_sampling(folder_path, output_csv):
  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  # Calcolo Frequenze Globali (L'Urna)
  fam_counts = {}
  clan_counts = {}
  total_n = 0
  
  for f in tqdm(files, desc="Calcolo frequenze dell'universo"):
    try:
      # Carichiamo solo le colonne necessarie
      df = pd.read_csv(
        f, sep='\t', comment='#', header=None, usecols=[5, 13], 
        names=['family', 'clan'], engine='c',
        na_values=['No_clan','\\N', ''] 
      )
      
      # Aggiornamento conteggi totali (Universo)
      total_n += len(df)
      
      # Conteggio famiglie
      for fam in df['family'].dropna():
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
      
      # Conteggio clan
      for cl in df['clan'].dropna():
        clan_counts[cl] = clan_counts.get(cl, 0) + 1
            
    except Exception as e:
      print(f"Errore nel file {f}: {e}")
      continue

  print(f"Universo analizzato: {total_n} domini totali.")
  
  # Trasformiamo i conteggi in vettori di probabilità (p_i)
  p_fam = np.array(list(fam_counts.values())) / total_n
  p_clan = np.array(list(clan_counts.values())) / total_n
  
  # Liberiamo memoria dai dizionari
  del fam_counts
  del clan_counts

  # Generazione Modello Analitico
  # Definiamo i checkpoint (taglie n) seguendo la crescita geometrica
  n_checkpoints = []
  curr_n = 10
  fattore_crescita = 1.1
  
  while curr_n <= total_n:
    n_checkpoints.append(curr_n)
    curr_n = int(curr_n * fattore_crescita) + 1
  
  # Aggiungiamo l'ultimo punto se non presente
  if n_checkpoints[-1] != total_n:
    n_checkpoints.append(total_n)
      
  risultati_null = []

  # Calcolo analitico tramite la formula del valore atteso
  # E[F(n)] = Sum_i ( 1 - (1 - p_i)^n )
  for n in tqdm(n_checkpoints, desc="Generazione modello analitico"):
    # Usiamo log1p per stabilità numerica: (1-p)^n = exp(n * log(1-p))
    expected_fam = np.sum(1 - np.exp(n * np.log1p(-p_fam)))
    expected_clan = np.sum(1 - np.exp(n * np.log1p(-p_clan)))
    
    risultati_null.append({
      'n': int(n),
      'Fn_family': int(round(expected_fam)),
      'Fn_clan': int(round(expected_clan))
    })
        
  # Salvataggio
  df_null = pd.DataFrame(risultati_null)
  df_null.to_csv(output_csv, index=False)
  
  print(f"Modello Analitico completato e salvato in {output_csv}")
  return df_null


def analizza_innovazione_per_clan(folder_path, output_file, top_n=5, clan_extra=None):
  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  # Identificazione dei Clan dominanti
  print("Identificazione dei Clan dominanti...")
  clan_counts = {}
  for f in tqdm(files, desc="Campionamento Clan"):
    try:
      df = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[13], names=['clan'], na_values=['No_clan','\\N', ''])
      df = df.dropna()
      for c in df['clan']:
        clan_counts[c] = clan_counts.get(c, 0) + 1
    except:
      continue

  # Prendiamo i top_n automatici
  auto_clans = sorted(clan_counts, key=clan_counts.get, reverse=True)[:top_n]
  print(f"Clan selezionati tra i top {top_n} per l'analisi:", auto_clans)
  
  # Uniamo con la tua lista manuale (usando set per evitare duplicati)
  if clan_extra is None:
    clan_extra = []
  
  # Lista finale dei Clan da analizzare
  top_clans = list(set(auto_clans + clan_extra))
  print(f"Clan totali selezionati {len(top_clans)}: {top_clans}")

  # Inizializzazione statistiche con checkpoint geometrico
  stats = {c: {
    'n': 0, 
    'fn': [], 
    'visti_fam': set(), 
    'steps': [], 
    'next_check': 1 # Iniziamo dal primo dominio trovato
  } for c in top_clans}
  
  # Analisi Per singolo Clan
  for f in tqdm(files, desc="Processing Clan Innovation"):
    try:
      df = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[5, 13], names=['family', 'clan'], na_values=['No_clan','\\N', ''])
    except:
      continue
    
    for c in top_clans:
      # Estraiamo le famiglie appartenenti al Clan corrente in questo file
      domini_clan = df[df['clan'] == c]['family'].values
      
      for fam in domini_clan:
        stats[c]['n'] += 1
        stats[c]['visti_fam'].add(fam)
        
        # Checkpoint Geometrico (logaritmico)
        if stats[c]['n'] >= stats[c]['next_check']:
          stats[c]['steps'].append(stats[c]['n'])
          stats[c]['fn'].append(len(stats[c]['visti_fam']))
          
          # Incremento del 10% (fattore 1.1)
          # Il +1 assicura che il valore cresca anche per n piccoli (es. 1, 2, 3...)
          stats[c]['next_check'] = int(stats[c]['next_check'] * 1.1) + 1


  # Generazione del file di output
  rows = []
  for clan_id, data in stats.items():
    for i in range(len(data['steps'])):
      rows.append({
        'clan': clan_id,
        'n_c': data['steps'][i],
        'fn_family': data['fn'][i]
      })
  
  df_output = pd.DataFrame(rows)
  df_output.to_csv(output_file, index=False)
  print(f"Dati salvati in {output_file}")
  return df_output


def mappa_phylum(folder_path, output_file, tax_file):
  # Creazione Mappa TaxID -> Phylum
  print(f"Caricamento tassonomia da {tax_file}...")
  tax_to_phylum = {}
  with open(tax_file, 'r', encoding='utf-8') as f:
    for line in f:
      parts = line.strip().split('\t')
      if len(parts) < 3: continue
      
      taxid = parts[0]
      lineage = parts[2]
      
      # Estrazione Phylum (solitamente il secondo elemento)
      hierarchy = lineage.split(';')
      if len(hierarchy) > 1:
        phylum = hierarchy[1]
      else:
        phylum = "Unclassified"
      
      tax_to_phylum[taxid] = phylum

  # Analisi dei nomi dei file e creazione lista per CSV
  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  phylum_groups = {} # Per il return (dizionario)
  rows_for_csv = []  # Per il file di output (lista di righe)
  
  for f in tqdm(files, desc="Mappatura Phylum"):
    # Estraiamo il TaxID (es. '41.tsv.gz' -> '41')
    taxid = f.name.split('.')[0]
    phylum = tax_to_phylum.get(taxid, "TaxID_Non_Trovato")
    
    # Aggiorniamo il dizionario per il return
    if phylum not in phylum_groups:
      phylum_groups[phylum] = []
    phylum_groups[phylum].append(f.name)
    
    # Aggiungiamo alla lista per il CSV
    rows_for_csv.append({
      "TaxID": taxid,
      "Phylum": phylum,
      "Filename": f.name
    })

  # Salvataggio su File CSV
  df_output = pd.DataFrame(rows_for_csv)
  df_output.to_csv(output_file, index=False)
  print(f"\nSalvataggio completato: {output_file}")

  return phylum_groups


if __name__ == "__main__":
  cartella_dati = "dati_proteomi" 
  nome_pfam = "risultati_heaps_innovazione.csv"
  nome_modello_a = "heaps_modello_a.csv"
  nome_clan = "heaps_clan.csv"
  nome_mappa_phylum = "mappa_phylum.csv"


  #--- Dati reali pfam ---
  # Controllo se rigenerare i dati
  my_file = Path(nome_pfam)
  if my_file.is_file():
    response = input(f"Il file {nome_pfam} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_dati_innovazione(cartella_dati, nome_pfam)
    else:
      df_heaps = pd.read_csv(nome_pfam)
  else:
    df_heaps = genera_dati_innovazione(cartella_dati, nome_pfam)


  #--- Dati per modello A (sampling) ---
  # Controllo se rigenerare i dati
  my_file_2 = Path(nome_modello_a)
  if my_file_2.is_file():
    response = input(f"Il file {nome_modello_a} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_modello_A_sampling(cartella_dati, nome_modello_a)
    else:
      df_heaps = pd.read_csv(nome_modello_a)
  else:
    df_heaps = genera_modello_A_sampling(cartella_dati, nome_modello_a)


  #--- Dati per studio relativo a specifici clan 

  # clan "speciali" da aggiungere
  clan_extra = ['CL0011', 'CL0016', 'CL0010']
  # Controllo se rigenerare i dati
  my_file_3 = Path(nome_clan)
  if my_file_3.is_file():
    response = input(f"Il file {nome_clan} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = analizza_innovazione_per_clan(cartella_dati, nome_clan, clan_extra=clan_extra)
    else:
      df_heaps = pd.read_csv(nome_clan)
  else:
    df_heaps = analizza_innovazione_per_clan(cartella_dati, nome_clan, clan_extra=clan_extra)
    

  #--- Mappa phylum per studio specifico innovazione

  # Controllo se rigenerare i dati
  file_tax = "ncbi_taxonomy.txt"
  my_file_4 = Path(nome_mappa_phylum)
  if my_file_4.is_file():
    response = input(f"Il file {nome_mappa_phylum} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = mappa_phylum(cartella_dati, nome_mappa_phylum, file_tax)
    else:
      df_heaps = pd.read_csv(nome_mappa_phylum)
  else:
    df_heaps = mappa_phylum(cartella_dati, nome_mappa_phylum, file_tax)

  