import os
import csv
import itertools
from Bio import SeqIO
from Bio.Seq import Seq

# --- CONFIGURATION ---

GENOMES_DIR = "/Users/arthur/M2_BIM/GENOM/GENOM_kmer/genomes"
K_SIZE = 4
OUTPUT_FILE = f"global_signatures_k{K_SIZE}.csv"

# --- FUNCTIONS ---

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


def calculate_signature(fasta_path, k, canonical_kmers):
    """
    Compute normalized canonical k-mer frequencies for a FASTA file.
    """
    kmer_counts = {kmer: 0 for kmer in canonical_kmers}
    total_valid_kmers = 0

    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence = str(record.seq).upper()

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
        print(f"  Error reading {fasta_path}: {e}")
        return None


# --- MAIN ---

def main():
    print(f"--- Starting k-mer analysis (k={K_SIZE}, canonical) ---")

    if not os.path.exists(GENOMES_DIR):
        print(f"Error: directory '{GENOMES_DIR}' does not exist.")
        return

    # Reference list of canonical k-mers
    canonical_kmers = generate_canonical_kmers(K_SIZE)

    with open(OUTPUT_FILE, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # CSV header
        header = ["Organism"] + canonical_kmers
        writer.writerow(header)

        processed = 0
        subdirs = sorted(os.listdir(GENOMES_DIR))

        for organism_dir in subdirs:
            full_path = os.path.join(GENOMES_DIR, organism_dir)

            if not os.path.isdir(full_path):
                continue

            fasta_path = os.path.join(full_path, "genome.fasta")

            if not os.path.exists(fasta_path):
                print(f"  Skipped {organism_dir} (genome.fasta not found)")
                continue

            print(f"Processing: {organism_dir}")

            signature = calculate_signature(
                fasta_path,
                K_SIZE,
                canonical_kmers
            )

            if signature:
                row = [organism_dir] + [
                    signature[kmer] for kmer in canonical_kmers
                ]
                writer.writerow(row)
                processed += 1

    print("\n--- Done ---")
    print(f"{processed} genomes processed")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
