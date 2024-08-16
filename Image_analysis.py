import numpy as np
import plotly.express as px
import umap
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from joblib import Parallel, delayed
from PIL import Image
import sqlite3

# Load reduced embeddings from a pickle file
def load_pca_embeddings(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Apply UMAP for dimensionality reduction with parallel processing
def apply_umap(embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', n_jobs=-1)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings, umap_model

# Interactive 3D plot with Plotly
def plotly_3d_umap(embeddings, labels, title):
    fig = px.scatter_3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        color=labels,
        title=title,
        labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
        color_continuous_scale='Spectral'
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()

# Example usage
pca_embeddings_path = "kernel_pca_embeddings_image_analysis.pkl"  # Path to the Kernel PCA-reduced embeddings
db_path = "image_metadata.db"  # Path to the SQLite database containing image paths

# Load the Kernel PCA-reduced embeddings
print("Loading Kernel PCA-reduced embeddings...")
embeddings_data = load_pca_embeddings(pca_embeddings_path)
uuids = list(embeddings_data.keys())
embeddings = np.array(list(embeddings_data.values()))
print("Kernel PCA-reduced embeddings loaded.")

# Visualize the data before clustering
print("Visualizing data with 3D UMAP before clustering...")
umap_embeddings, umap_model = apply_umap(embeddings, n_components=3, n_neighbors=15, min_dist=0.01)

# Apply K-Means clustering
n_clusters = 16  # Adjust the number of clusters as needed
print(f"Applying K-Means clustering with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(embeddings)

# Interactive 3D plot for K-Means clustering
plotly_3d_umap(umap_embeddings, kmeans_labels, 'Interactive 3D UMAP with K-Means Clustering')

# Apply Agglomerative Clustering in 3D
print(f"Applying Agglomerative Clustering with {n_clusters} clusters...")
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(embeddings)

# Interactive 3D plot for Agglomerative Clustering
plotly_3d_umap(umap_embeddings, agg_labels, f'Interactive 3D UMAP with Agglomerative Clustering (n_clusters={n_clusters})')
