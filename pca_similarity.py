import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

def save_to_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def incremental_pca_fit(embedding_matrix, n_components, batch_size):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for i in tqdm(range(0, len(embedding_matrix), batch_size), desc="Fitting IncrementalPCA"):
        ipca.partial_fit(embedding_matrix[i:i + batch_size])
    return ipca

def incremental_pca_transform(embedding_matrix, ipca, batch_size):
    reduced_embeddings = []
    for i in tqdm(range(0, len(embedding_matrix), batch_size), desc="Transforming Embeddings"):
        reduced_embeddings.append(ipca.transform(embedding_matrix[i:i + batch_size]))
    return np.vstack(reduced_embeddings)

def apply_incremental_pca_and_save(embeddings_pickle_path, pca_pickle_path, pca_model_path, n_components=50, batch_size=10000):
    print("Applying Incremental PCA...")
    embeddings = load_from_pickle(embeddings_pickle_path)
    image_uuids, embedding_matrix = zip(*embeddings.items())
    embedding_matrix = np.array(embedding_matrix)

    # Fit IncrementalPCA
    ipca = incremental_pca_fit(embedding_matrix, n_components, batch_size)

    # Save PCA model
    with open(pca_model_path, 'wb') as f:
        pickle.dump(ipca, f)
    
    # Transform embeddings using IncrementalPCA
    reduced_embeddings = incremental_pca_transform(embedding_matrix, ipca, batch_size)

    # Create a dictionary to store reduced embeddings with their UUIDs
    pca_data = {uuid: embedding for uuid, embedding in zip(image_uuids, reduced_embeddings)}

    # Save the reduced embeddings to a pickle file
    save_to_pickle(pca_data, pca_pickle_path)
    print(f"PCA reduced embeddings saved to {pca_pickle_path}")

if __name__ == "__main__":
    embeddings_pickle_path = "combined_embeddings.pkl"  # Path to your existing embeddings
    pca_pickle_path = "pca_embeddings.pkl"  # Path where the reduced embeddings will be saved
    pca_model_path = "pca_model.pkl"  # Path to save the PCA model

    # Apply Incremental PCA and save the reduced embeddings and PCA model
    apply_incremental_pca_and_save(embeddings_pickle_path, pca_pickle_path, pca_model_path, n_components=50, batch_size=10000)
