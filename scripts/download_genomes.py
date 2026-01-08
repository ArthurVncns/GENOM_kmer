import os
import gzip
import shutil
import time
import urllib.request
from Bio import Entrez

BASE_DIR = "/Users/arthur/M2_BIM/GENOM/GENOM_kmer"

# Required by NCBI Entrez
Entrez.email = "random@gmail.com"

# Read organism names (one per line)
with open(os.path.join(BASE_DIR, "genomes_labeled.txt"), "r") as f:
    organisms = [line.strip().split(",")[0] for line in f if line.strip()]

def download_genome_and_cds(organism_name):
    """
    Download the reference genome and CDS. 
    Deletes the directory if any step fails.
    """
    print(f"Processing: {organism_name}")
    output_dir = None  # Initialize variable for scope safety

    try:
        # --- 1. Search for Genome ---
        handle = Entrez.esearch(db="assembly", term=organism_name, retmax=1)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            print("-> Genome not found")
            return False

        assembly_id = record["IdList"][0]

        # Retrieve assembly summary
        handle = Entrez.esummary(db="assembly", id=assembly_id)
        summary = Entrez.read(handle)
        handle.close()

        doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
        ftp_url = doc["FtpPath_RefSeq"] or doc["FtpPath_GenBank"]
        
        if not ftp_url:
            print("  -> No FTP path available")
            return False

        label = os.path.basename(ftp_url)
        genome_url = f"{ftp_url}/{label}_genomic.fna.gz"

        # --- 2. Create Directory ---
        organism_dir = organism_name.replace(" ", "_")
        output_dir = os.path.join(BASE_DIR, "genomes", organism_dir)
        os.makedirs(output_dir, exist_ok=True)

        genome_fasta = os.path.join(output_dir, "genome.fasta")
        cds_fasta = os.path.join(output_dir, "cds.fasta")

        # --- 3. Download Genome ---
        try:
            with urllib.request.urlopen(genome_url) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    with open(genome_fasta, "wb") as out:
                        shutil.copyfileobj(gz, out)
            print("  -> Genome downloaded")
        except Exception as e:
            raise Exception(f"Genome download failed: {e}")

        # --- 4. Search and Download CDS ---
        search_term = (
            f"{organism_name}[Organism] "
            "AND biomol_genomic[PROP] "
            "AND RefSeq[filter]"
        )

        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=1)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            print("  -> CDS not found (Cleaning up...)")
            # Manual cleanup because this isn't an Exception, just missing data
            shutil.rmtree(output_dir)
            return False

        nucleotide_id = record["IdList"][0]

        handle = Entrez.efetch(
            db="nucleotide",
            id=nucleotide_id,
            rettype="fasta_cds_na",
            retmode="text"
        )
        cds_data = handle.read()
        handle.close()

        with open(cds_fasta, "w") as f:
            f.write(cds_data)

        print("  -> CDS downloaded")
        return True

    except Exception as error:
        print(f"  -> Error: {error}")
        # Clean up directory if it was created and an error occurred
        if output_dir and os.path.exists(output_dir):
            print(f"  -> Removing incomplete directory: {output_dir}")
            shutil.rmtree(output_dir)
        return False

def main():
    for organism in organisms:
        success = download_genome_and_cds(organism)
#        if success:
#            time.sleep(1)  # Respect NCBI rate limits

    print("\nDownload completed.")


if __name__ == "__main__":
    main()
