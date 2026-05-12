import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import numpy as np


def mappa_phylum(folder_path, output_file, tax_file):
  """
  Associa ogni proteoma al rispettivo Phylum di appartenenza utilizzando la tassonomia GTDB.
  Questa funzione esegue il parsing del file di tassonomia per mappare i TaxID 
  alle gerarchie filogenetiche. Filtra i file presenti nel dataset per creare 
  un'istantanea (snapshot) che colleghi i nomi dei file ai gruppi tassonomici.
  """
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


def genera_heaps_proteomi(folder_path, mappa_phylum_csv, output_csv="heaps_proteomi.csv"):
  """
  Estrae le metriche di scaling (n, F, C) per ogni organismo inteso come sistema indipendente.
  Per ogni proteoma nel dataset, la funzione calcola la taglia totale (n), il numero 
  di famiglie distinte (F) e il patrimonio di Clan (C).
  """
  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  # Carichiamo la mappatura Phylum fatta in precedenza
  df_map = pd.read_csv(mappa_phylum_csv)
  # Creiamo un dizionario per lookup veloce {Filename: Phylum}
  dict_map = pd.Series(df_map.Phylum.values, index=df_map.Filename).to_dict()

  data = []
  for f in tqdm(files, desc="Estrazione Proteomi"):
    try:
      # Carichiamo il file completo
      df = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[5, 13], names=['fam', 'clan'], na_values=['No_clan', '\\N', ''])
      
      data.append({
        'Filename': f.name,
        'Phylum': dict_map.get(f.name, "Unknown"),
        'n_total': len(df),
        'F_total': df['fam'].nunique(),
        'C_total': df['clan'].nunique()
      })
    except: continue

  df_snapshot = pd.DataFrame(data)
  df_snapshot.to_csv(output_csv, index=False)
  return df_snapshot


def genera_urna(folder_path, output_csv="frequenze_famiglie.csv", min_size=100):
  """
  Genera l'urna di campionamento stocastico per la validazione del modello nullo.
  Costruisce una distribuzione di frequenze globali delle famiglie Pfam campionando 
  esclusivamente da proteomi che superano la soglia di qualità e taglia minima. 
  L'urna definisce le probabilità p_i necessarie per quantificare il residuo relativo 
  e la saturazione funzionale.
  """

  path = Path(folder_path)
  files = list(path.glob("*.tsv.gz"))
  
  global_counts = Counter()
  total_domini = 0
  conteggio_esclusi = 0
  
  for f in tqdm(files, desc="Generazione dell'Urna"):
    try:
      # Leggiamo il file
      df = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[5], names=['family'])
      
      # FILTRO: Se il proteoma è troppo piccolo, non lo usiamo per l'Urna
      if len(df) < min_size:
        conteggio_esclusi += 1
        continue
        
      global_counts.update(df['family'].dropna())
      total_domini += len(df)
    except: 
      continue
  
  # Creiamo il DF delle probabilità
  df_urna = pd.DataFrame.from_dict(global_counts, orient='index', columns=['count'])
  df_urna['p_i'] = df_urna['count'] / total_domini
  df_urna.to_csv(output_csv)
  
  print(f"Urna creata con {len(df_urna)} famiglie diverse.")
  print(f"Domini totali: {total_domini} | Proteomi esclusi (<{min_size}): {conteggio_esclusi}")


def genera_urna_top6(folder_path, mappa_phylum_csv, output_csv="frequenze_famiglie_top6.csv", min_size=100):
  """
  Genera l'urna campionando solo dai proteomi appartenenti ai primi 6 Phyla 
  e con taglia superiore a min_size.
  """
  path = Path(folder_path)
  
  # Identificazione dei Top 6 Phyla dai metadati
  df_mappa = pd.read_csv(mappa_phylum_csv)
  top_phyla = df_mappa['Phylum'].value_counts().nlargest(6).index.tolist()
  
  # Filtriamo i metadati per ottenere solo i nomi dei file dei Phyla target
  file_target = set(df_mappa[df_mappa['Phylum'].isin(top_phyla)]['Filename'].tolist())
  
  global_counts = Counter()
  total_domini = 0
  conteggio_letti = 0
  conteggio_esclusi_taglia = 0
  
  # Scansione dei file
  files = list(path.glob("*.tsv.gz"))
  
  for f in tqdm(files, desc="Generazione Urna (Top 6 Phyla)"):
    # Controlliamo se il file appartiene ai Phyla selezionati
    if f.name not in file_target:
      continue
        
    try:
      # Leggiamo la colonna delle famiglie (Pfams)
      df = pd.read_csv(f, sep='\t', comment='#', header=None, usecols=[5], names=['family'])
      
      # Filtro taglia minima (n_total)
      if len(df) < min_size:
        conteggio_esclusi_taglia += 1
        continue
          
      global_counts.update(df['family'].dropna())
      total_domini += len(df)
      conteggio_letti += 1
    except Exception as e:
      print(f"Errore nel file {f.name}: {e}")
      continue
  
  # Calcolo probabilità p_i
  df_urna = pd.DataFrame.from_dict(global_counts, orient='index', columns=['count'])
  df_urna['p_i'] = df_urna['count'] / total_domini
  df_urna.to_csv(output_csv)
  
  print(f"\n--- Riepilogo Urna ---")
  print(f"Phyla inclusi: {top_phyla}")
  print(f"Proteomi validi letti: {conteggio_letti}")
  print(f"Proteomi scartati per taglia (<{min_size}): {conteggio_esclusi_taglia}")
  print(f"Famiglie Pfam totali nell'urna: {len(df_urna)}")
  print(f"Domini totali (N): {total_domini}")
  
  return df_urna


def genera_modello_nullo(dati_reali_csv, urna_csv, iterations=100, output_csv="modello_nullo.csv"):
  """
  Simula il modello nullo tramite campionamento Monte Carlo.
  Per ogni proteoma di taglia n, esegue 'iterations' campionamenti 
  per calcolare media e variabilità (deviazione standard).
  """
  print(f"Caricamento dati e inizializzazione ({iterations} iterazioni)...")
  df_reale = pd.read_csv(dati_reali_csv)
  df_urna = pd.read_csv(urna_csv)
  
  # Probabilità e indici delle famiglie nell'universo Pfam
  p_i = df_urna['p_i'].values
  famiglie_indices = np.arange(len(p_i))
  
  # Filtro taglia minima
  df_reale = df_reale[df_reale['n_total'] > 100]

  def monte_carlo_sampling(n):
    # Eseguiamo n_iterazioni di campionamento casuale per una taglia n
    risultati_iterazioni = []
    for _ in range(iterations):
      # Pesca n elementi basandoti sulle probabilità p_i
      sample = np.random.choice(famiglie_indices, size=int(n), p=p_i, replace=True)
      # Conta quante famiglie distinte sono state pescate
      risultati_iterazioni.append(len(np.unique(sample)))
    
    # Restituiamo media e deviazione standard
    return np.mean(risultati_iterazioni), np.std(risultati_iterazioni)

  # Attiviamo tqdm per i DataFrame di Pandas
  tqdm.pandas(desc="Simulazione Monte Carlo")
  
  print(f"Analisi di {len(df_reale)} proteomi in corso...")
  
  # Applichiamo la simulazione con progress bar
  stats = df_reale['n_total'].progress_apply(monte_carlo_sampling)
  
  # Distribuiamo i risultati nelle colonne del dataset
  df_reale['F_null_mean'] = stats.apply(lambda x: x[0])
  df_reale['F_null_std'] = stats.apply(lambda x: x[1])
  
  # Calcolo residuo relativo basato sulla media delle simulazioni
  # Formula: (F_reale - F_null_mean) / F_null_mean
  df_reale['residuo_relativo'] = (df_reale['F_total'] - df_reale['F_null_mean']) / df_reale['F_null_mean']

  df_reale.to_csv(output_csv, index=False)
  print(f"\nSalvataggio completato: {output_csv}")
  return df_reale


def genera_modello_nullo_binnato(dati_reali_csv, urna_csv, iterations=1000, n_bins=18, output_csv="modello_nullo_binnato.csv"):
  """
  Genera un ensemble di realizzazioni del modello nullo (matrici intere)
  e calcola la variabilità sulle medie binnate.
  """
  print(f"Caricamento dati e inizializzazione ({iterations} matrici sintetiche)...")
  df_reale = pd.read_csv(dati_reali_csv)
  df_urna = pd.read_csv(urna_csv)

  # Selezione Top 6 Phylum
  top_phyla = df_reale['Phylum'].value_counts().nlargest(6).index
  df_top = df_reale[df_reale['Phylum'].isin(top_phyla)].copy().sort_values('n_total')
  
  p_i = df_urna['p_i'].values
  famiglie_indices = np.arange(len(p_i))
  
  # Filtro e Creazione Bin logaritmici sulla taglia n reale
  df_top = df_top[df_top['n_total'] > 100].sort_values('n_total')
  # Crea bin equispaziati in scala LOGARITMICA tra 300 e il massimo
  bins = np.logspace(np.log10(300), np.log10(df_top['n_total'].max()), num=n_bins)
  df_top['bin'] = pd.cut(df_top['n_total'], bins=bins, labels=False, include_lowest=True)
  
  # Calcolo centri dei bin e media reale dell'unica osservazione
  res_reale = df_top.groupby('bin').agg({
    'n_total': 'mean',
    'F_total': 'mean'
  }).rename(columns={'F_total': 'F_reale_mean', 'n_total': 'n_bin_center'})

  # Generazione dell'Ensemble (Matrici intere)
  # Stoccheremo le medie binnate di ogni realizzazione
  matrice_medie_simulate = []

  print(f"Generazione Ensemble su {len(df_top)} proteomi...")
  for i in tqdm(range(iterations), desc="Realizzazioni Ensemble"):
    f_simulati = []
    # Per ogni proteoma reale, generiamo una "copia" stocastica
    for n in df_top['n_total']:
      sample = np.random.choice(famiglie_indices, size=int(n), p=p_i, replace=True)
      f_simulati.append(len(np.unique(sample)))
    
    # Creiamo la curva media per questa specifica realizzazione
    df_temp = pd.DataFrame({'bin': df_top['bin'], 'F_sim': f_simulati})
    media_binnata_sim = df_temp.groupby('bin')['F_sim'].mean().values
    matrice_medie_simulate.append(media_binnata_sim)

  # Analisi Statistica dell'Ensemble
  matrice_medie_simulate = np.array(matrice_medie_simulate)
  
  # Media delle medie e Deviazione Standard delle medie (Variabilità del modello)
  res_reale['F_null_mean'] = np.mean(matrice_medie_simulate, axis=0)
  res_reale['F_null_std'] = np.std(matrice_medie_simulate, axis=0)
  
  # Residuo basato sui trend medi
  res_reale['residuo_relativo'] = (res_reale['F_reale_mean'] - res_reale['F_null_mean']) / res_reale['F_null_mean']

  res_reale.to_csv(output_csv)
  print(f"\nSalvataggio completato: {output_csv}")
  return res_reale


if __name__ == "__main__":
  cartella_dati = "dati_proteomi" 
  output_heaps = "heaps_proteomi.csv"
  output_mappa_phylum = "mappa_phylum.csv"
  output_urna = "frequenze_famiglie.csv"
  output_urna_top6 = "frequenze_famiglie_top6.csv"
  output_modello_nullo = "modello_nullo.csv"
  output_modello_nullo_binnato = "modello_nullo_binnato.csv"


  #--- Mappa phylum - proteomi ---
  file_tax = "ncbi_taxonomy.txt"
  # Controllo se rigenerare i dati
  my_file_4 = Path(output_mappa_phylum)
  if my_file_4.is_file():
    response = input(f"Il file {output_mappa_phylum} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = mappa_phylum(cartella_dati, output_mappa_phylum, file_tax)
    else:
      df_heaps = pd.read_csv(output_mappa_phylum)
  else:
    df_heaps = mappa_phylum(cartella_dati, output_mappa_phylum, file_tax)


  #--- Innovazione e legge di Heaps ---
  # Controllo se rigenerare i dati
  my_file = Path(output_heaps)
  if my_file.is_file():
    response = input(f"Il file {output_heaps} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_heaps_proteomi(cartella_dati, output_mappa_phylum, output_heaps)
    else:
      df_heaps = pd.read_csv(output_heaps)
  else:
    df_heaps = genera_heaps_proteomi(cartella_dati, output_mappa_phylum, output_heaps)

  
  #--- Urna ---
  # Controllo se rigenerare i dati
  my_file = Path(output_urna)
  if my_file.is_file():
    response = input(f"Il file {output_urna} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_urna(cartella_dati,  output_urna)
    else:
      df_heaps = pd.read_csv(output_urna)
  else:
    df_heaps = genera_urna(cartella_dati, output_urna)


  #--- Urna top6 ---
  # Controllo se rigenerare i dati
  my_file = Path(output_urna_top6)
  if my_file.is_file():
    response = input(f"Il file {output_urna_top6} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_urna_top6(cartella_dati, output_mappa_phylum,  output_urna_top6)
    else:
      df_heaps = pd.read_csv(output_urna_top6)
  else:
    df_heaps = genera_urna_top6(cartella_dati, output_mappa_phylum, output_urna_top6)

  
  #--- Modello nullo ---
  # Controllo se rigenerare i dati
  my_file = Path(output_modello_nullo)
  if my_file.is_file():
    response = input(f"Il file {output_modello_nullo} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_modello_nullo(output_heaps, output_urna, 100 ,output_modello_nullo)
    else:
      df_heaps = pd.read_csv(output_modello_nullo)
  else:
    df_heaps = genera_modello_nullo(output_heaps, output_urna, 100, output_modello_nullo)


  #--- Modello nullo binnato ---
  # Controllo se rigenerare i dati
  my_file = Path(output_modello_nullo_binnato)
  if my_file.is_file():
    response = input(f"Il file {output_modello_nullo_binnato} esiste. Rigenerare? (y/n): ")
    if response.lower() == 'y':
      df_heaps = genera_modello_nullo_binnato(dati_reali_csv=output_heaps, urna_csv=output_urna_top6, output_csv=output_modello_nullo_binnato)
    else:
      df_heaps = pd.read_csv(output_modello_nullo_binnato)
  else:
    df_heaps = genera_modello_nullo_binnato(dati_reali_csv=output_heaps, urna_csv=output_urna_top6, output_csv=output_modello_nullo_binnato)