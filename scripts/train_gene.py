import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from model import GenomicClassifier, EarlyStopping
from gene_assignement import generate_canonical_kmers, calculate_signature_gene

K = 4
GENOME_DIR = "genomes"
MIN_SEQ_LEN = 300
MAX_GENES_PER_GENOME = 1500
MIN_GENES_COUNT = 30
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def main():
    canonical_kmers = generate_canonical_kmers(K)
    organism_dirs = [d for d in os.listdir(GENOME_DIR)]

    X = []
    y = []

    print("Retrieving data...")
    for organism in tqdm.tqdm(organism_dirs, desc="Organisms"):
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

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_train_losses = []

    print(f"\nStarting {n_splits}-fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_encoded)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split data
        X_train_fold = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        y_train_fold = torch.tensor(y_encoded[train_idx], dtype=torch.long)
        X_val_fold = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
        y_val_fold = torch.tensor(y_encoded[val_idx], dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=BATCH_SIZE, shuffle=True)

        model = GenomicClassifier(X_scaled.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        fold_losses = []

        early_stopper = EarlyStopping(patience=5, min_delta=0.001)
        
        for epoch in range(EPOCHS):
            # TRAINING
            model.train()
            epoch_loss = 0
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
            
            for batch_X, batch_y in pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            fold_losses.append(avg_train_loss)

            # VALIDATION (early stopping)
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold.to(device))
                val_loss = criterion(val_outputs, y_val_fold.to(device)).item()
            
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Final validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold.to(device))
            _, preds = torch.max(val_outputs, 1)
            fold_acc = accuracy_score(y_val_fold.cpu(), preds.cpu())
            print(f"Fold {fold+1} Validation Accuracy: {fold_acc:.4f}")
            fold_results.append(fold_acc)
            all_train_losses.append(fold_losses)

    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(all_train_losses):
        plt.plot(losses, label=f'Fold {i+1}')
    
    plt.title(f"Training Loss over {EPOCHS} Epochs (K-Fold CV)")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True)
    
    loss_path = os.path.join(FIG_DIR, "nn_gene_loss.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"CV Average Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
    print(f"Figure saved in : {loss_path}")