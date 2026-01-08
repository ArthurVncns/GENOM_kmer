import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib.patches as mpatches

# --- CONFIGURATION ---

SIGNATURE_FILE = "global_signatures_k4.csv"
LABEL_FILE = "genomes_labeled.txt"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# --- 1. LOAD DATA ---

df = pd.read_csv(SIGNATURE_FILE)
print(f"Signature matrix shape: {df.shape}")

labels = pd.read_csv(LABEL_FILE, names=["Organism", "Domain"])
labels['Organism'] = labels['Organism'].str.replace(" ", "_")

df = df.merge(labels, on="Organism", how="inner")

organism_names = df["Organism"].values
domains = df["Domain"].values
X = df.drop(columns=["Organism", "Domain"])

# --- 2. NORMALIZATION (POUR PCA) ---

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PCA ---

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var * 100:.2f}%")

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Organism"] = organism_names
pca_df["Domain"] = domains

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Domain", s=80, alpha=0.8)

plt.title("PCA of genomic k-mer signatures (k=4)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_k4.png"), dpi=300)
plt.close()

# --- 4. HIERARCHICAL CLUSTERING ---

distance_matrix = cosine_distances(X)
condensed_dist = squareform(distance_matrix)
linked = linkage(condensed_dist, method="average")

plt.figure(figsize=(12, 10))

# 1. Définir une palette de couleurs
domain_colors = {
    "Bacteria": "#1f77b4",  # Bleu
    "Archaea": "#d62728"    # Rouge
    # Ajoutez "Eukaryota": "green" si besoin
}

# 2. Créer un mapping : Organisme -> Couleur
# On associe chaque nom d'organisme à son domaine pour retrouver la couleur vite
org_to_domain = dict(zip(organism_names, domains))

# 3. Tracer le dendrogramme
d = dendrogram(
    linked,
    labels=organism_names,
    orientation="top",
    leaf_rotation=90,
    leaf_font_size=10
)

# 4. Appliquer les couleurs aux labels de l'axe X
ax = plt.gca() # Récupère l'axe courant
x_labels = ax.get_xmajorticklabels() # Récupère la liste des objets "texte" en bas

for label in x_labels:
    org_name = label.get_text()
    # On cherche le domaine de cet organisme
    domain = org_to_domain.get(org_name)
    
    if domain in domain_colors:
        label.set_color(domain_colors[domain])
        # Optionnel : mettre en gras pour mieux voir
        label.set_fontweight("bold") 

# 5. Ajouter une légende manuelle (car le dendrogramme ne le fait pas)
legend_patches = [
    mpatches.Patch(color=color, label=domain) 
    for domain, color in domain_colors.items()
]
plt.legend(handles=legend_patches, title="Domain", loc='upper right')

plt.title("Hierarchical clustering (Colored by Domain)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "dendrogram_k4_colored.png"), dpi=300)
plt.close()

print("\nAnalysis completed.")
print(f"Figures saved in '{FIG_DIR}/'")