import numpy as np
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import dtale as dt
from pathlib import Path
import os

from script import analizza_prognosi_innovazione

# --- CARICAMENTO DATI ---
file_path = "risultati_heaps_innovazione.csv"
if not os.path.exists(file_path):
  raise FileNotFoundError(f"Assicurati di aver generato il file {file_path}")

df_heaps = pd.read_csv(file_path)

# Impostiamo lo stile generale dei plot
plt.style.use('bmh')
plt.rc('font', family='serif', serif='Times New Roman')
# Diciamo a Matplotlib di usare un font compatibile con la matematica
plt.rcParams['mathtext.fontset'] = 'cm'
# impostiamo la grandezza dei font degli oggetti principali
plt.rcParams.update({'font.size': 12})        # Font base
plt.rcParams.update({'axes.titlesize': 11})   # Titoli dei grafici
plt.rcParams.update({'axes.labelsize': 14})   # Label dei grafici
plt.rcParams.update({'legend.fontsize': 12})   # Legenda
plt.rcParams.update({'lines.linewidth': 1.8})   # Larghezza linee
plt.rcParams.update({'lines.linestyle': 'solid'})   # Larghezza linee
plt.rcParams.update({'lines.markersize': 5})  # Grandezza marker

# ---- 1) Ci interessa sapere come varia l'innovazione con la taglia. La probabilità di innovazione è costante per sistemi che hanno taglie diverse? 

# Regressioni (Per trovare beta)
# Usiamo i logaritmi per trovare la pendenza della retta nel piano log-log
def get_beta_innovazione(x, y):
  mask = (x > 0) & (y > 0) & (df_heaps['n'] < 1e4)
  #mask = (df_heaps['n'] > 10000) & (df_heaps['n'] < 1000000) & (x > 0) & (y > 0) & (df_heaps['n'] < 1e4)
  log_x = np.log10(x[mask])
  log_y = np.log10(y[mask])
  slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
  return slope, intercept, r_value

def get_beta_plateau(x, y):
  mask = (x > 0) & (y > 0) & (df_heaps['n'] > 1e4)
  #mask = (df_heaps['n'] > 10000) & (df_heaps['n'] < 1000000) & (x > 0) & (y > 0) & (df_heaps['n'] < 1e4)
  log_x = np.log10(x[mask])
  log_y = np.log10(y[mask])
  slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
  return slope, intercept, r_value

beta_fam, intercept_fam, r_fam = get_beta_innovazione(df_heaps['n'], df_heaps['Fn_family'])
beta_clan, intercept_clan, r_clan = get_beta_innovazione(df_heaps['n'], df_heaps['Fn_clan'])
beta_fam_p, intercept_fam_p, r_fam_p = get_beta_plateau(df_heaps['n'], df_heaps['Fn_family'])
beta_clan_p, intercept_clan_p, r_clan_p = get_beta_plateau(df_heaps['n'], df_heaps['Fn_clan'])

# Plot
width  = 6
height = width / 1.618
fig1, (ax1, bx1) = plt.subplots(2, 1, figsize=(width, height + 4))
plt.subplots_adjust(hspace=0.5)

# Plot 1: Famiglie
ax1.loglog(df_heaps['n'], df_heaps['Fn_family'], 'o', color='C0', alpha=0.5, label='Dati Pfam (Famiglie)')
# Linea di fit innovazione
fit_fam = 10**intercept_fam * (df_heaps['n']**beta_fam)
ax1.loglog(df_heaps['n'], fit_fam, color='black', linestyle='--', 
           label=fr'Fit: $\beta \approx {beta_fam:.3f}$ ($R^2={r_fam**2:.3f}$)')

# Linea di fit plateau
fit_fam_p = 10**intercept_fam_p * (df_heaps['n']**beta_fam_p)
ax1.loglog(df_heaps['n'], fit_fam_p, color='C3', linestyle='--', 
           label=fr'Fit: $\beta \approx {beta_fam_p:.3f}$ ($R^2={r_fam_p**2:.3f}$)')

#ax1.set_xlabel('$n$')
ax1.set_ylabel(r'$F_{fam}$')
ax1.set_ylim(top=1e5)
ax1.legend(loc='lower right')
ax1.grid(True, which="both", ls="-", alpha=0.3)

# Plot 2: Clan
bx1.loglog(df_heaps['n'], df_heaps['Fn_clan'], 'd', color='C1', alpha=0.5, label='Dati Pfam (Clan)')
# Linea di fit
fit_clan = 10**intercept_clan * (df_heaps['n']**beta_clan)
bx1.loglog(df_heaps['n'], fit_clan, color='black', linestyle='--', 
           label=fr'Fit: $\beta \approx {beta_clan:.3f}$ ($R^2={r_clan**2:.3f}$)')

# Linea di fit plateau
fit_clan_p = 10**intercept_clan_p * (df_heaps['n']**beta_clan_p)
bx1.loglog(df_heaps['n'], fit_clan_p, color='C3', linestyle='--', 
           label=fr'Fit: $\beta \approx {beta_clan_p:.3f}$ ($R^2={r_clan_p**2:.3f}$)')

bx1.set_xlabel('Taglia cumulativa $n$')
bx1.set_ylabel(r'$F_{clan}$')
bx1.set_ylim(top=1e5)
bx1.legend(loc='lower right')
bx1.grid(True, which="both", ls="-", alpha=0.3)

# salvo il plot in PDF
plt.tight_layout()  # Evita che i titoli o le etichette vengano tagliati
cartella_salvataggio = Path(r"C:\Users\calci\OneDrive\Desktop\Lab\Computazionale\DC2\latex\plot")
percorso_finale = cartella_salvataggio / f'Heaps_innovazione.pdf'
plt.savefig(percorso_finale, format='pdf', bbox_inches='tight')
print(f"Salvato con successo in: {percorso_finale}")

print(f"Esponente Famiglie innovazione: {beta_fam:.4f}")
print(f"Esponente Clan innovazione: {beta_clan:.4f}")
print(f"Esponente Famiglie plateau: {beta_fam_p:.4f}")
print(f"Esponente Clan plateau: {beta_clan_p:.4f}")


# ---- 2) Ci interessa sapere se la diversità è prognostica di innovazione, cioè se sistemi che hanno un patrimonio più ricco di moduli di classi diverse hanno più tendenza a innovare.
cartella_dati = "dati_proteomi" 
nome_output = "risultati_heaps_innovazione.csv"
nome_modello_a = "heaps_modello_a.csv"

df_reale = pd.read_csv(nome_output)
df_null = pd.read_csv(nome_modello_a)

# Funzione interna per calcolare il tasso di innovazione
def calcola_rate(df):
  df = df.copy()
  df['dn'] = df['n'].diff()
  df['df_fam'] = df['Fn_family'].diff()
  df['innovation_rate'] = df['df_fam'] / df['dn']
  return df.dropna()

df_p_reale = calcola_rate(df_reale)
df_p_null = calcola_rate(df_null)

width  = 6
height = width / 1.618
fig, ax2 = plt.subplots(figsize=(width, height))

# --- DATI REALI (Scatter con Colore logaritmico) ---
scatter = ax2.scatter(
  df_p_reale['Fn_clan'], 
  df_p_reale['innovation_rate'], 
  c=df_p_reale['n'], 
  norm=LogNorm(), 
  cmap='viridis', 
  marker='H',
  alpha=0.6, 
  edgecolors='none',
  s=40,
  label='Dati Reali (Pfam)'
)

# --- MODELLO A (Linea Rossa di confronto) ---
# Usiamo una media mobile o una linea per il modello nullo per pulizia visiva
ax2.plot(
  df_p_null['Fn_clan'], 
  df_p_null['innovation_rate'], 
  color='red', 
  linestyle='--', 
  linewidth=2, 
  alpha=0.8,
  label='Modello A (Sampling)'
)

# Impostazioni Assi
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Patrimonio di Clan ($F_{clan}$)')
ax2.set_ylabel(r'Tasso di Innovazione ($\Delta F_{fam} / \Delta n$)')
ax2.grid(True, which="both", ls="-", alpha=0.2)

# Legenda e Colorbar
ax2.legend(loc='lower left')
cbar = fig.colorbar(scatter, ax=ax2)
cbar.set_label('Taglia cumulativa $n$')

# Salvataggio
plt.tight_layout()
cartella_salvataggio = Path(r"C:\Users\calci\OneDrive\Desktop\Lab\Computazionale\DC2\latex\plot")
cartella_salvataggio.mkdir(parents=True, exist_ok=True)
percorso_finale = cartella_salvataggio / 'Prognosi_innovazione_confronto.pdf'
plt.savefig(percorso_finale, format='pdf', bbox_inches='tight')
print(f"Salvato con successo in: {percorso_finale}")

# D-Tale per ispezione finale
d = dt.show(df_reale, host='localhost')
# d.open_browser() # Decommenta se vuoi aprirlo subito

plt.show()