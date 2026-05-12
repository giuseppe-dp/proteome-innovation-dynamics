import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress, spearmanr

import dtale as dt

# --- Stile Plot---
plt.style.use('bmh')
plt.rc('font', family='serif', serif='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({
  'font.size': 12,
  'axes.titlesize': 14,
  'axes.labelsize': 14,
  'legend.fontsize': 11,
  'lines.linewidth': 1.5,
  'lines.linestyle': 'solid',
  'lines.markersize': 5
})

# Caricamento e Filtro
df = pd.read_csv("heaps_proteomi.csv")
df = df[df['n_total'] > 100]  # Filtro per proteomi troppo piccoli

# Selezione Top 6 Phylum
top_phyla = df['Phylum'].value_counts().nlargest(6).index
df_top = df[df['Phylum'].isin(top_phyla)].copy().sort_values('n_total')
df_top['P_innovazione'] = df_top['F_total'] / df_top['n_total']

# Creazione Figura a due Pannelli
width = 6
height =(width / 1.618) + 4
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height))
fig.subplots_adjust(hspace=0.7)

# 1) SCALING DELL'INNOVAZIONE (Domanda 1)
sns.scatterplot(data=df_top, x='n_total', y='P_innovazione', 
                hue='Phylum', alpha=0.6, palette='Set1', edgecolor='none', ax=ax1)

ax1.set_xscale('log')
ax1.set_yscale('log')

# Media globale
media_globale = df['F_total'].sum() / df['n_total'].sum()
ax1.axhline(media_globale, color='black', linestyle='--', label=f'Media Globale: {media_globale:.3f}')

ax1.set_xlabel(r"Taglia del Proteoma ($n$)")
ax1.set_ylabel(r"Densità di Innovazione P ($F/n$)")
ax1.grid(True, which="both", alpha=0.3)

# 2) PROGNOSI DELL'INNOVAZIONE (Domanda 2)
sns.scatterplot(data=df_top, x='C_total', y='F_total', 
                hue='Phylum', alpha=0.6, palette='Set1', edgecolor='none', ax=ax2)

ax2.set_xscale('log')
ax2.set_yscale('log')

# Preparazione dati per il Fit Log-Log
# Usiamo i logaritmi in base 10 per coerenza con gli assi del grafico
log_c = np.log10(df_top['C_total'])
log_f = np.log10(df_top['F_total'])

# Esecuzione del Fit Lineare
slope, intercept, r_value, p_value, std_err = linregress(log_c, log_f)

# Creazione della retta di regressione per il plot
# Generiamo dei punti X nello spazio dei logaritmi per coprire l'estensione dei dati
x_fit = np.linspace(log_c.min(), log_c.max(), 100)
y_fit = slope * x_fit + intercept

# Aggiunta della retta al grafico ax2
# Dobbiamo riportare x e y alla scala originale (10^) perché l'asse è logaritmico
ax2.plot(10**x_fit, 10**y_fit, color='black', linestyle='-', alpha=0.7)

# Calcolo correlazione
corr_coef, p_val = spearmanr(df_top['C_total'], df_top['F_total'])
stats_text = (f'Spearman $R$: {corr_coef:.3f}, $p$-val < 0.001\n'
              f'Power Law $\\alpha$: {slope:.3f} $\\pm${std_err:.3f}')

ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='none', alpha=0.5, edgecolor='none'))


ax2.set_xlabel(r"Clan Distinti ($C$)")
ax2.set_ylabel(r"Famiglie Distinte ($F$)")
ax2.grid(True, which="both", alpha=0.3)

# --- Gestione delle legende ---
# Prendi tutti gli handles e labels da ax1
handles, labels = ax1.get_legend_handles_labels()

h_phyla, l_phyla = [], []
h_line, l_line = [], []

# Filtriamo in base al contenuto del testo
for h, l in zip(handles, labels):
  if "Media Globale" in l:
    h_line.append(h)
    l_line.append(l)
  elif l != "Phylum": # Escludiamo il titolo della hue che Seaborn aggiunge
    h_phyla.append(h)
    l_phyla.append(l)

# Rimuovo le legende automatiche
if ax1.get_legend(): ax1.get_legend().remove()
if ax2.get_legend(): ax2.get_legend().remove()

# Legenda Media Globale (interna ad ax1)
ax1.legend(h_line, l_line, loc='lower left', frameon=False)

# Legenda Phylum (esterna su ax2 o sulla figura)
ax2.legend(h_phyla, l_phyla, loc='upper left', title="Top 6 Phyla")


# --- Salvataggio ---
plt.tight_layout()
cartella_salvataggio = Path(r"C:\Users\calci\OneDrive\Desktop\Lab\Computazionale\DC2\latex\plot")
cartella_salvataggio.mkdir(parents=True, exist_ok=True)
percorso_finale = cartella_salvataggio / 'Confronto_Innovazione_Prognosi.pdf'

plt.savefig(percorso_finale, format='pdf', bbox_inches='tight')
print(f"Grafico salvato in: {percorso_finale}")


#--- Modello Nullo vs Reale ---
# Caricamento dei dati binnati
df_binned = pd.read_csv("modello_nullo_binnato.csv")

fig4, (dx1, dx2) = plt.subplots(2, 1, figsize=(width, height))
fig4.subplots_adjust(hspace=0.5)

# --- PLOT 1 CONFRONTO TRA ENSEMBLE E OSSERVAZIONE SPERIMENTALE ---
# In scala log-log per visualizzare la Legge di Heaps

# Area di variabilità dell'ensemble nullo (2 sigma sulla media binnata)
dx1.fill_between(
  df_binned['n_bin_center'], 
  df_binned['F_null_mean'] - 2 * df_binned['F_null_std'],
  df_binned['F_null_mean'] + 2 * df_binned['F_null_std'],
  color='gray', alpha=0.5, label=r'Incertezza ($2\sigma$)'
)

# Trend medio del Modello Nullo (la media di tutte le matrici simulate)
dx1.plot(df_binned['n_bin_center'], df_binned['F_null_mean'], 
         color='black', linestyle='--', linewidth=2, label='Aspettativa Nulla')

# L'unica osservazione sperimentale (Dati Reali)
dx1.plot(df_binned['n_bin_center'], df_binned['F_reale_mean'], 
         color='red', marker='o', markersize=5, linewidth=2.5, label='Osservazione Reale')

dx1.set_xscale('log')
dx1.set_yscale('log')
dx1.set_xlabel(r"Taglia del Proteoma ($n$)")
dx1.set_ylabel(r"Media Famiglie Distinte ($\langle F \rangle$)")
dx1.grid(True, which="both", alpha=0.2)
dx1.legend(loc='upper left')

# --- PLOT 2 ANALISI DEI RESIDUI SULLA MEDIA ---
# Mostra la deviazione della traiettoria reale dall'ensemble nullo

# Incertezza relativa del modello nullo (2 sigma / media)
incertezza_relativa = (2 * df_binned['F_null_std']) / df_binned['F_null_mean']

dx2.fill_between(
  df_binned['n_bin_center'], 
  -incertezza_relativa, 
  incertezza_relativa, 
  color='gray', alpha=0.5, label=r'Incertezza ($2\sigma$)'
)

# Residuo della traiettoria reale
dx2.plot(df_binned['n_bin_center'], df_binned['residuo_relativo'], 
         color='red', marker='o', markersize=5, linewidth=2.5, label='Trend Residuo')

dx2.axhline(0, color='black', linestyle='--', linewidth=1.5)
dx2.set_xscale('log')
dx2.set_xlabel(r"Taglia del Proteoma ($n$)")
dx2.set_ylabel(r"Residuo Relativo $\Delta R$")
dx2.grid(True, which="both", alpha=0.2)
dx2.legend(loc='lower left')

plt.tight_layout()
percorso_finale = cartella_salvataggio / "Trend_modello_nullo.pdf"
plt.savefig(percorso_finale, format='pdf', bbox_inches='tight')
print(f"Grafico salvato in: {percorso_finale}")

# d = dt.show(df_binned, host='localhost') # Se vuoi ispezionare i dati filtrati

plt.show()