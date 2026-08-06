
"""
Exploration initiale du dataset CIC-IDS 2017.

Charge les 8 fichiers CSV, les concatène et affiche un résumé :
- nombre de lignes et de colonnes
- distribution des classes
- valeurs manquantes et infinies
- problèmes dans les noms de colonnes

Correspond à l'étape 1 du framework Awad et al. (2025).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# Chemins des dossiers
DATA_DIR = Path("data/raw")
LOG_DIR = Path("results/logs")


# Lister les fichiers CSV
print("Recherche des fichiers CSV dans", DATA_DIR)
csv_files = sorted(DATA_DIR.glob("*.csv"))

for f in csv_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}  ({size_mb:.1f} MB)")

print(f"Total : {len(csv_files)} fichiers\n")


# Charger et concaténer tous les fichiers en un seul DataFrame
print("Chargement des fichiers...")
dataframes = []

for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"  {f.name} : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)
print(f"\nDataFrame total : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")


# Analyser les noms de colonnes
# Problème connu du dataset CIC-IDS 2017 : espaces au début/fin des noms
print("Analyse des colonnes")
print(f"  Nombre de colonnes : {len(df.columns)}")

cols_avec_espaces = [c for c in df.columns if c != c.strip()]
print(f"  Colonnes avec espaces parasites : {len(cols_avec_espaces)}")
if cols_avec_espaces:
    print("  Exemples :")
    for c in cols_avec_espaces[:3]:
        print(f"    '{c}'")

# Identifier la colonne de labels
label_col = df.columns[-1]  # elle est toujours la dernière
print(f"  Colonne de labels : '{label_col}'\n")


# Distribution des classes (déséquilibre)
print("Distribution des classes")
counts = df[label_col].value_counts()
total = len(df)

print(f"  Nombre de classes : {len(counts)}")
print(f"  Nombre total de records : {total:,}\n")

for classe, nombre in counts.items():
    pourcentage = 100 * nombre / total
    print(f"  {str(classe):<40s} {nombre:>12,}  ({pourcentage:.3f}%)")

# Ratio de déséquilibre
ratio = counts.max() / counts.min()
print(f"\n  Ratio de déséquilibre (max/min) : {ratio:,.0f} pour 1\n")


# Valeurs manquantes (NaN)
print("Valeurs manquantes et infinies")

na_counts = df.isna().sum()
na_counts = na_counts[na_counts > 0]

if len(na_counts) == 0:
    print("  Aucune valeur manquante")
else:
    print(f"  Colonnes avec des NaN : {len(na_counts)}")
    for col, count in na_counts.items():
        pct = 100 * count / len(df)
        print(f"    {col} : {count:,} ({pct:.3f}%)")


# Valeurs infinies (uniquement dans les colonnes numériques)
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
inf_counts = inf_counts[inf_counts > 0]

if len(inf_counts) == 0:
    print("  Aucune valeur infinie")
else:
    print(f"  Colonnes avec des Inf : {len(inf_counts)}")
    for col, count in inf_counts.items():
        pct = 100 * count / len(df)
        print(f"    {col} : {count:,} ({pct:.3f}%)")


# Sauvegarder un résumé rapide dans results/logs/
LOG_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"exploration_{timestamp}.txt"

with open(log_path, "w") as f:
    f.write(f"Exploration du dataset CIC-IDS 2017\n")
    f.write(f"Date : {datetime.now()}\n\n")
    f.write(f"Nombre de fichiers CSV : {len(csv_files)}\n")
    f.write(f"Nombre total de lignes : {total:,}\n")
    f.write(f"Nombre de colonnes : {len(df.columns)}\n")
    f.write(f"Nombre de classes : {len(counts)}\n")
    f.write(f"Ratio de deséquilibre : {ratio:,.0f} pour 1\n\n")
    f.write("Distribution des classes :\n")
    for classe, nombre in counts.items():
        pct = 100 * nombre / total
        f.write(f"  {classe} : {nombre:,} ({pct:.3f}%)\n")

print(f"\nRapport sauvegardé dans : {log_path}")