# Gene Classification Using Genomic Signatures (k-mers)

This project was carried out as part of the **Master's program in Bioinformatics and Modeling**. It aims to predict the genomic origin of coding sequences (CDS) using **k-mer composition (k = 4)** as a biological signature.

## 🎯 Objectives
- Extract and process genomic signatures from prokaryotic genomes (RefSeq).
- Compare the performance of classical statistical distance metrics with machine learning models.
- Develop a high-performance classifier capable of assigning a gene to one of **87 reference organisms**.

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
Supervised learning models significantly outperform simple distance-based methods:

| Model | Mean Accuracy |
| :--- | :--- |
| **Distance (Correlation)** | ~66.6% |
| **Random Forest** | 74.6% |
| **XGBoost** | 77.4% |
| **Deep Learning (MLP)** | **85.6% (± 0.23)** |

*Note: A specific case study (*Ca. Cloacamonas acidaminovorans*) showed 0% accuracy using distance-based methods, suggesting strong genomic heterogeneity or horizontal gene transfer events.*

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

# Training and evaluation of the neural network
python scripts/nn_genomes.py
```
