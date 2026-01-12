import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from gene_assignement import generate_all_kmers, canonical_kmer, generate_canonical_kmers, calculate_signature_gene

K = 4
GENOME_DIR = "genomes"
LABEL_FILE = "genomes_labeled.txt"
MAX_GENES_PER_GENOME = 1000
MIN_SEQ_LEN = 300
MIN_GENES_COUNT = 30

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    canonical_kmers = generate_canonical_kmers(K)
    organism_dirs = [d for d in os.listdir(GENOME_DIR)]

    labels_df = pd.read_csv(LABEL_FILE, names=["Organism", "Domain"])
    labels_df['Organism'] = labels_df['Organism'].str.replace(" ", "_")
    labels_dict = dict(zip(labels_df["Organism"], labels_df["Domain"]))

    X = []
    y = []

    print("Retrieving data...")
    for organism in organism_dirs:
        genes_file = os.path.join(GENOME_DIR, organism, "cds.fasta")
        if not os.path.exists(genes_file):
            continue

        genes = list(SeqIO.parse(genes_file, "fasta"))
        valid_genes = [g for g in genes if len(g.seq) >= MIN_SEQ_LEN]

        if len(valid_genes) < MIN_GENES_COUNT:
            continue

        if len(valid_genes) > MAX_GENES_PER_GENOME:
            valid_genes = random.sample(valid_genes, MAX_GENES_PER_GENOME)

        for gene in valid_genes:
            seq = str(gene.seq).upper()
            
            signature_dict = calculate_signature_gene(seq, K, canonical_kmers)
            
            if signature_dict is not None:
                signature_vector = [signature_dict[kmer] for kmer in canonical_kmers]
                X.append(signature_vector)
                y.append(labels_dict[organism])
    
    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {X.shape[0]} genes.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (Linear)": SVC(kernel='linear', random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', random_state=42)
    }

    plt.figure(figsize=(15, 5))

    for i, (name, clf) in enumerate(models.items()):
        print(f"\nEvaluating {name}...")
        
        y_pred_idx = cross_val_predict(clf, X, y_encoded, cv=cv, n_jobs=-1)
        
        acc = accuracy_score(y_encoded, y_pred_idx)
        print(f"Accuracy {name}: {acc:.4f}")

        ax = plt.subplot(1, 3, i + 1)
        cm = confusion_matrix(y_encoded, y_pred_idx)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f"{name}\nAcc: {acc:.2f}")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "binary_classification_domains.png"), dpi=300)

    print(f"\nDone. Confusion matrix saved in: {FIG_DIR}")

if __name__ == "__main__":
    main()