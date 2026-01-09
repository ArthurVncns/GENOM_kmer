import os
import itertools
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.spatial.distance import cdist

K = 4
CSV_FILE = "global_signatures_k4.csv"
GENOME_DIR = "genomes"
MIN_SEQ_LEN = 300
MIN_GENES_COUNT = 30


def generate_all_kmers(k):
    """
    Generate all possible k-mers (A,C,G,T) in lexicographic order.
    """
    bases = ["A", "C", "G", "T"]
    return [''.join(p) for p in itertools.product(bases, repeat=k)]


def canonical_kmer(kmer):
    """
    Return the canonical form of a k-mer:
    min(kmer, reverse_complement(kmer))
    """
    rc = str(Seq(kmer).reverse_complement())
    return min(kmer, rc)


def generate_canonical_kmers(k):
    """
    Generate a sorted list of unique canonical k-mers.
    """
    canonical_set = set()
    for kmer in generate_all_kmers(k):
        canonical_set.add(canonical_kmer(kmer))
    return sorted(canonical_set)


def calculate_signature_gene(sequence, k, canonical_kmers):
    """
    Compute normalized canonical k-mer frequencies for a sequence.
    """
    sequence = sequence.upper()
    kmer_counts = {kmer: 0 for kmer in canonical_kmers}
    total_valid_kmers = 0

    try:
        for i in range(len(sequence) - k + 1):
            word = sequence[i:i + k]
            if set(word) <= {"A", "C", "G", "T"}:
                canon = canonical_kmer(word)
                kmer_counts[canon] += 1
                total_valid_kmers += 1
        if total_valid_kmers == 0:
                return None

        # Normalize counts into frequencies
        return {
            kmer: count / total_valid_kmers
            for kmer, count in kmer_counts.items()
        }

    except Exception as e:
        print(f"Error reading sequence: {e}")
        return None


def main():
    df_ref = pd.read_csv(CSV_FILE, index_col=0, header=0)

    reference_matrix = df_ref.values
    reference_labels = df_ref.index.values

    canonical_kmers = generate_canonical_kmers(K)

    organism_dirs = [d for d in os.listdir(GENOME_DIR)]
    gene_vectors = []
    true_labels = []
    
    print("Reading genes and computing genomic signatures")

    for organism in organism_dirs:
        if organism not in reference_labels:
            continue
            
        fasta_path = os.path.join(GENOME_DIR, organism, "cds.fasta")
        if not os.path.exists(fasta_path):
            continue
        
        genes = list(SeqIO.parse(fasta_path, "fasta"))

        valid_genes = [g for g in genes if len(g.seq) >= MIN_SEQ_LEN]

        if len(valid_genes) < MIN_GENES_COUNT:
            continue

        for gene in valid_genes:
            seq_str = str(gene.seq)

            vec = calculate_signature_gene(seq_str, K, canonical_kmers)

            if vec is not None:
                vec_ordered = [vec[k] for k in canonical_kmers]
                gene_vectors.append(vec_ordered)
                true_labels.append(organism)
        

    print(f"\nData : {len(gene_vectors)} genes to classify")
    
    X_genes = np.array(gene_vectors)
    y_true = np.array(true_labels)

    metrics_list = ['cosine', 'euclidean', 'cityblock', 'braycurtis', 'correlation']
    
    all_genomes = np.unique(y_true)
    df_final_results = pd.DataFrame(index=all_genomes)
    
    print("Computing distances and classification")

    for metric in metrics_list:

        print(f"Using {metric} distance...")

        try:
            dists = cdist(X_genes, reference_matrix, metric=metric)
            closest_indices = np.argmin(dists, axis=1)

            y_pred = reference_labels[closest_indices]
            
            df_temp = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
            
            df_temp['Correct'] = df_temp['Actual'] == df_temp['Predicted']
            accuracy_per_genome = df_temp.groupby('Actual')['Correct'].mean()
            df_final_results[metric] = accuracy_per_genome

        except Exception as e:
            print(f"Error with metric {metric}: {e}")

    global_averages = df_final_results.mean()
    df_final_results.loc['TOTAL_MEAN'] = global_averages
    output_filename = "results/dist_accuracy_per_genome.csv"
    df_final_results.to_csv(output_filename)
    
    print(f"Averages : \n{global_averages}")

if __name__ == "__main__":
    main()