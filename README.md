# Gene Classification Using Genomic Signatures (k-mers)

This project was carried out as part of the **Master's program in Bioinformatics and Modeling**. It aims to predict the genomic origin of coding sequences (CDS) using **k-mer composition (k = 4)** as a biological signature.

## 🎯 Objectives
- Extract and process genomic signatures from prokaryotic genomes (RefSeq).
- Compare the performance of classical statistical distance metrics with machine learning models.
- Develop a high-performance classifier capable of assigning a gene or a genome fragment to one of **81 reference organisms**.

## 🛠️ Technical Pipeline
1. **Acquisition**  
   Automated download of genomes and CDS sequences via NCBI Entrez.

2. **Processing**  
   Computation of **canonical 4-mer frequencies** (136-dimensional feature space).

3. **Exploratory Analysis**  
   Visualization of genomic space using **PCA** and **Hierarchical Clustering** (Scipy / Seaborn).

4. **Classification**  
   - Distance-based methods (Correlation, Cosine, Euclidean).
   - Machine Learning models (Random Forest, XGBoost).
   - Deep Learning approach (MLP neural network).

## 📈 Results
Supervised learning models significantly outperform simple distance-based methods. We also observed a massive performance boost when shifting from gene-specific classification to genome-wide fragment classification.

### 🧬 Gene-level Classification (CDS)
| Model | Mean Accuracy |
| :--- | :--- |
| **Distance (Correlation)** | ~66.6% |
| **Random Forest** | 74.6% |
| **XGBoost** | 77.4% |
| **Deep Learning (MLP)** | **85.6% (± 0.23)** |

### 🌍 Metagenomic-like Classification (5kb Genome Chunks)
To simulate real-world environmental sampling, we trained the MLP on 5,000 bp random genome fragments. 
- **Deep Learning (MLP) Accuracy: 98.59% (± 0.0016)**
  
> **Key Finding:** While individual genes can be tricky to classify due to small sample sizes and evolutionary pressure (85.6%), random 5kb genomic fragments provide a much more stable and nearly perfect taxonomic signal (98.6%), proving the reliability of k-mer signatures for metagenomic identification.

*Note: A specific case study (*Ca. Cloacamonas acidaminovorans*) showed 0% accuracy using distance-based methods on genes, but was successfully recovered by the Deep Learning model (90.5%).*

## 💻 Installation & Usage

### Dependencies
- Python 3.12+
- Biopython  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- PyTorch  
- Matplotlib, Seaborn  

### Execution
```bash
# Download data
python scripts/download_genomes.py

# Training the neural network on genome fragments
python scripts/train_chunk.py
```
