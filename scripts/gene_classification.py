import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import product
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from gene_assignement import generate_all_kmers, canonical_kmer, generate_canonical_kmers, calculate_signature_gene

# ---------------------
# 1. CONFIGURATION
# ---------------------
K = 4  # taille du k-mer
ROOT_DIR = "genomes"
LABEL_FILE = "genomes_labeled.txt"
MIN_SEQ_LEN = 300
MIN_GENES_COUNT = 30

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------
# 2. GENERATE CANONICAL K-MERS
# ---------------------
def canonical_kmer(kmer):
    rc = str(Seq(kmer).reverse_complement())
    return min(kmer, rc)

# All possible canonical k-mers
all_kmers = [''.join(p) for p in product('ATCG', repeat=K)]
canonical_set = sorted(set(canonical_kmer(k) for k in all_kmers))

# ---------------------
# 3. K-MER COUNTS
# ---------------------
def get_kmer_counts(seq, k, canonical_kmers):
    seq = seq.upper()
    counts = {kmer: 0 for kmer in canonical_kmers}
    if len(seq) < k: return [0]*len(canonical_kmers)
    
    for i in range(len(seq) - k + 1):
        word = seq[i:i+k]
        if set(word) <= {'A','T','C','G'}:
            canon = canonical_kmer(word)
            counts[canon] += 1
    
    total = sum(counts.values())
    if total == 0: return [0]*len(canonical_kmers)
    return [counts[kmer]/total for kmer in canonical_kmers]

# ---------------------
# 4. LOAD LABELS
# ---------------------
labels_df = pd.read_csv(LABEL_FILE, names=["Organism", "Domain"])
labels_df['Organism'] = labels_df['Organism'].str.replace(" ", "_")
labels_dict = dict(zip(labels_df["Organism"], labels_df["Domain"]))

# ---------------------
# 5. BUILD DATASET
# ---------------------
X = []
y = []
groups = []  # for GroupShuffleSplit

print("Loading genes and computing k-mer signatures...")

for organism in os.listdir(ROOT_DIR):
    genes_file = os.path.join(ROOT_DIR, organism, "cds.fasta")
    if not os.path.exists(genes_file):
        continue

    genes = list(SeqIO.parse(genes_file, "fasta"))

    valid_genes = [g for g in genes if len(g.seq) >= MIN_SEQ_LEN]

    if len(valid_genes) < MIN_GENES_COUNT:
        continue

    for gene in genes:
        seq = str(gene.seq).upper()
        X.append(get_kmer_counts(seq, K, canonical_set))
        y.append(labels_dict[organism])
        groups.append(organism)

X = np.array(X)
y = np.array(y)
groups = np.array(groups)

print(f"Dataset ready: {X.shape[0]} genes, {len(set(groups))} genomes, {len(set(y))} domains")

# ---------------------
# 6. SPLIT TRAIN / TEST BY GENOME
# ---------------------
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ---------------------
# 7. TRAIN CLASSIFIER
# ---------------------
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ---------------------
# 8. EVALUATE
# ---------------------
y_pred = clf.predict(X_test)
print("\nClassification report (per domain):")
print(classification_report(y_test, y_pred))

acc = np.mean(y_pred == y_test)
print(f"Overall accuracy: {acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(cm, display_labels=np.unique(y))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Gene Classification")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

print(f"\nFigures saved in '{FIG_DIR}/'")
print("Script completed successfully.")
