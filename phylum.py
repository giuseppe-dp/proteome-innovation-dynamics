import pandas as pd

#--- Piccolo script per vedere tutti i phylum presenti nel dataset ---
# Caricamento del file creato in precedenza
df = pd.read_csv('mappa_phylum.csv')

# Conteggio dei TaxID per ogni Phylum
# Usiamo 'nunique' per sicurezza, ma 'count' va bene se ogni riga è un TaxID unico
report_phylum = df.groupby('Phylum')['TaxID'].count().reset_index()

# Rinominiamo la colonna per chiarezza
report_phylum.columns = ['Phylum', 'Conteggio_TaxID']

# Ordiniamo dal Phylum più popoloso
report_phylum = report_phylum.sort_values(by='Conteggio_TaxID', ascending=False)

# Visualizzazione
print(report_phylum.to_string(index=True))

# (Opzionale) Salva il report
# report_phylum.to_csv('report_distribuzione_phylum.csv', index=False)

# Caricamento del dataset
df = pd.read_csv("heaps_proteomi.csv")

# Lista dei 6 Phyla selezionati dalle tue note
top_6 = ['Proteobacteria', 'Actinobacteria', 'Firmicutes', 'Bacteroidetes', 'Cyanobacteria', 'Tenericutes']

# Filtraggio e aggregazione
tabella_dati = df[df['Phylum'].isin(top_6)].groupby('Phylum').agg({
  'Phylum': 'count',      # Numero di proteomi
  'n_total': 'sum',       # Somma totale dei domini n
  'F_total': 'sum',      # Media delle famiglie (o 'sum' se preferisci il totale)
  'C_total': 'sum'       # Media dei clan (o 'sum')
}).rename(columns={'Phylum': 'N_Proteomi'}).reindex(top_6)

# Calcolo dei totali per la riga finale
tabella_dati.loc['Totale'] = tabella_dati.sum()

print(tabella_dati.round(2))