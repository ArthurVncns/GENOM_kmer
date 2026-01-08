import os
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------
# CONFIGURATION
# ---------------------

K = 4
ROOT_DIR = "genomes"
MAX_GENES_PER_GENOME = 500
MIN_SEQ_LEN = 200
RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)

# ---------------------
# FONCTIONS K-MER
# ---------------------

def canonical_kmer(kmer):
    rc = str(Seq(kmer).reverse_complement())
    return min(kmer, rc)

all_kmers = [''.join(p) for p in product('ATCG', repeat=K)]
canonical_set = sorted(set(canonical_kmer(k) for k in all_kmers))

def get_kmer_counts(seq, k, canonical_kmers):
    seq = seq.upper()
    counts = {kmer: 0 for kmer in canonical_kmers}
    if len(seq) < k: return [0]*len(canonical_kmers)
    for i in range(len(seq) - k + 1):
        word = seq[i:i+k]
        if set(word) <= {'A','T','C','G'}:
            counts[canonical_kmer(word)] += 1
    total = sum(counts.values())
    if total == 0: return [0]*len(canonical_kmers)
    return [counts[kmer]/total for kmer in canonical_kmers]

# ---------------------
# CHARGEMENT DES DONNÉES
# ---------------------

X = []
y = []

print("Loading genes...")
organisms_list = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

for organism in organisms_list:
    genes_file = os.path.join(ROOT_DIR, organism, "cds.fasta")
    if not os.path.exists(genes_file):
        continue

    genes = list(SeqIO.parse(genes_file, "fasta"))
    
    # Downsampling
    if len(genes) > MAX_GENES_PER_GENOME:
        genes = random.sample(genes, MAX_GENES_PER_GENOME)

    for gene in genes:
        seq = str(gene.seq).upper()
        if len(seq) < MIN_SEQ_LEN or 'N' in seq:
            continue
        
        X.append(get_kmer_counts(seq, K, canonical_set))
        y.append(organism)

X = np.array(X)
y = np.array(y)

print(f"Dataset: {X.shape[0]} genes from {len(set(y))} organisms.")

# ---------------------
# SPLIT
# ---------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# ---------------------
# ENTRAÎNEMENT
# ---------------------
print("Training RandomForest...")
clf = RandomForestClassifier(
    n_estimators=100, 
    n_jobs=-1, 
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train, y_train)

# ---------------------
# 8. ÉVALUATION & SAUVEGARDE CSV
# ---------------------

print("\nPredicting on test set...")
y_pred = clf.predict(X_test)

print("Generating classification report...")
report_dict = classification_report(y_test, y_pred, output_dict=True)

df_report = pd.DataFrame(report_dict).transpose()
df_metrics = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
df_global = df_report.loc[['accuracy', 'macro avg', 'weighted avg'], :]

metrics_path = os.path.join(RES_DIR, "classification_metrics.csv")
df_metrics.index.name = "Organism"
df_metrics.to_csv(metrics_path)
print(f"-> Per-genome metrics saved to: {metrics_path}")

print("Generating confusion analysis...")
confusion_data = []
classes = sorted(list(set(y)))
df_conf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
confusion_counts = df_conf.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
confusion_counts['Total_Actual'] = confusion_counts.groupby('Actual')['Count'].transform('sum')
confusion_counts['Percentage'] = (confusion_counts['Count'] / confusion_counts['Total_Actual']) * 100
confusion_counts = confusion_counts.sort_values(by=['Actual', 'Count'], ascending=[True, False])

conf_path = os.path.join(RES_DIR, "confusion_details.csv")
confusion_counts.to_csv(conf_path, index=False, float_format="%.2f")
print(f"-> Confusion details saved to: {conf_path}")