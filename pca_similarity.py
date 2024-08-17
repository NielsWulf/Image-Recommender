import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.utils import shuffle
from tqdm import tqdm


def save_to_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}


def incremental_pca_fit(embedding_matrix, n_components, batch_size):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for i in tqdm(
        range(0, len(embedding_matrix), batch_size), desc="Fitting IncrementalPCA"
    ):
        ipca.partial_fit(embedding_matrix[i : i + batch_size])
    return ipca


def incremental_pca_transform(embedding_matrix, ipca, batch_size):
    reduced_embeddings = []
    for i in tqdm(
        range(0, len(embedding_matrix), batch_size), desc="Transforming Embeddings"
    ):
        reduced_embeddings.append(ipca.transform(embedding_matrix[i : i + batch_size]))
    return np.vstack(reduced_embeddings)


def apply_kernel_pca(embedding_matrix, n_components, kernel="rbf"):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, n_jobs=-1)
    reduced_embeddings = kpca.fit_transform(embedding_matrix)
    return kpca, reduced_embeddings


def subsample_data(embeddings, uuids, sample_size):
    subsampled_uuids, subsampled_embeddings = shuffle(
        uuids, embeddings, n_samples=sample_size, random_state=42
    )
    return subsampled_uuids, np.array(subsampled_embeddings)


def apply_incremental_and_kernel_pca_and_save(
    embeddings_pickle_path,
    pca_pickle_path,
    pca_model_path,
    kernel_pca_pickle_path,
    n_components_ipca=50,
    n_components_kpca=100,
    batch_size=10000,
    kernel_pca_sample_size=10000,
):
    print("Applying Incremental PCA...")
    embeddings = load_from_pickle(embeddings_pickle_path)
    image_uuids, embedding_matrix = zip(*embeddings.items())
    embedding_matrix = np.array(embedding_matrix)

    # Step 1: Apply Incremental PCA
    ipca = incremental_pca_fit(embedding_matrix, n_components_ipca, batch_size)

    # Save Incremental PCA model
    with open(pca_model_path, "wb") as f:
        pickle.dump(ipca, f)

    # Transform embeddings using Incremental PCA
    reduced_embeddings_ipca = incremental_pca_transform(
        embedding_matrix, ipca, batch_size
    )

    # Save the Incremental PCA reduced embeddings
    ipca_data = {
        uuid: embedding for uuid, embedding in zip(image_uuids, reduced_embeddings_ipca)
    }
    save_to_pickle(ipca_data, pca_pickle_path)
    print(f"Incremental PCA reduced embeddings saved to {pca_pickle_path}")

    # Step 2: Subsample Data for Kernel PCA
    print(f"Subsampling data to {kernel_pca_sample_size} samples for Kernel PCA...")
    subsampled_uuids, subsampled_embeddings = subsample_data(
        reduced_embeddings_ipca, image_uuids, kernel_pca_sample_size
    )

    # Step 3: Apply Kernel PCA on the subsampled data
    print("Applying Kernel PCA...")
    kpca, reduced_embeddings_kpca = apply_kernel_pca(
        subsampled_embeddings, n_components=n_components_kpca
    )

    # Save Kernel PCA model
    with open(kernel_pca_pickle_path, "wb") as f:
        pickle.dump(kpca, f)

    # Save the Kernel PCA reduced embeddings
    kpca_data = {
        uuid: embedding
        for uuid, embedding in zip(subsampled_uuids, reduced_embeddings_kpca)
    }
    save_to_pickle(kpca_data, kernel_pca_pickle_path)
    print(f"Kernel PCA reduced embeddings saved to {kernel_pca_pickle_path}")


if __name__ == "__main__":
    embeddings_pickle_path = (
        "combined_embeddings.pkl"  # Path to your existing embeddings
    )
    pca_pickle_path = "pca_embeddings_image_analysis_50.pkl"  # Path where the Incremental PCA reduced embeddings will be saved
    pca_model_path = (
        "pca_model_image_analysis_50.pkl"  # Path to save the Incremental PCA model
    )
    kernel_pca_pickle_path = "kernel_pca_embeddings_image_analysis.pkl"  # Path to save the Kernel PCA reduced embeddings

    # Apply Incremental PCA followed by Kernel PCA and save the reduced embeddings and models
    apply_incremental_and_kernel_pca_and_save(
        embeddings_pickle_path,
        pca_pickle_path,
        pca_model_path,
        kernel_pca_pickle_path,
        n_components_ipca=50,  # Number of components for Incremental PCA
        n_components_kpca=100,  # Number of components for Kernel PCA
        batch_size=10000,  # Batch size for Incremental PCA
        kernel_pca_sample_size=10000,  # Number of samples for Kernel PCA
    )
