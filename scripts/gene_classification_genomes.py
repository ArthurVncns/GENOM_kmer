import os
import random
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from gene_assignement import generate_all_kmers, canonical_kmer, generate_canonical_kmers, calculate_signature_gene


K = 4
GENOME_DIR = "genomes"
MAX_GENES_PER_GENOME = 800
MIN_SEQ_LEN = 300
MIN_GENES_COUNT = 30
RES_DIR = "results"
os.makedirs(RES_DIR, exist_ok=True)

def main():
    canonical_kmers = generate_canonical_kmers(K)
    organism_dirs = [d for d in os.listdir(GENOME_DIR)]

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
                y.append(organism)
    
    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {X.shape[0]} genes from {len(set(y))} organisms.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        tree_method='hist', 
        random_state=42
    )

    print("Performing Random Forest...")
    y_pred_rf_idx = cross_val_predict(rf, X, y_encoded, cv=cv, n_jobs=-1)
    print("Performing XGBoost...")
    y_pred_xgb_idx = cross_val_predict(xgb, X, y_encoded, cv=cv, n_jobs=-1)

    y_pred_rf = le.inverse_transform(y_pred_rf_idx)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_idx)
    
    print("Preparing and saving results...")
    results = []
    for org in le.classes_:
        mask = (y == org)
        
        acc_rf = np.mean(y_pred_rf[mask] == y[mask])
        acc_xgb = np.mean(y_pred_xgb[mask] == y[mask])
        count = np.sum(mask)
        
        results.append({
            'Organism': org,
            'Gene_Count': count,
            'Accuracy_RF': acc_rf,
            'Accuracy_XGB': acc_xgb,
            'Best_Model': 'RF' if acc_rf > acc_xgb else 'XGB' if acc_xgb > acc_rf else 'Tie'
        })

    df_res = pd.DataFrame(results)
    total_mean = pd.Series({
        'Organism': 'TOTAL_MEAN',
        'Gene_Count': df_res['Gene_Count'].sum(),
        'Accuracy_RF': df_res['Accuracy_RF'].mean(),
        'Accuracy_XGB': df_res['Accuracy_XGB'].mean(),
        'Best_Model': '-'
    })
    df_res = pd.concat([df_res, total_mean.to_frame().T], ignore_index=True)

    csv_path = os.path.join(RES_DIR, "comparaison_rf_xgboost.csv")
    df_res.to_csv(csv_path, index=False)

    print(f"\nResults saved in: {csv_path}")
    print(f"Mean RF: {total_mean['Accuracy_RF']:.4f}")
    print(f"Mean XGB: {total_mean['Accuracy_XGB']:.4f}")

if __name__ == "__main__":
    main()