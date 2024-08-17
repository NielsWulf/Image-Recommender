import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
from torchvision import models, transforms
from PIL import Image
import pickle

# Load pre-trained ResNet model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
model.eval()

# Define image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_embeddings_in_batches(file_path, batch_size):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    uuids = list(data.keys())
    embeddings = list(data.values())
    for i in range(0, len(embeddings), batch_size):
        yield uuids[i : i + batch_size], embeddings[i : i + batch_size]


def compute_embedding(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(img_tensor)
    embedding = embedding.view(-1).numpy()  # Flatten the embedding
    return embedding


def compute_multiple_embeddings(img_paths, model):
    embeddings = []
    for img_path in img_paths:
        embedding = compute_embedding(img_path, model)
        embeddings.append(embedding)
    return np.mean(embeddings, axis=0)  # Average the embeddings


def find_top_similar_images(embedding, embeddings_batches, pca, top_n=5):
    embedding_pca = pca.transform([embedding])
    all_similarities = []
    all_uuids = []

    for uuids_batch, embeddings_batch in embeddings_batches:
        similarities = cosine_similarity(embedding_pca, embeddings_batch)[0]
        all_similarities.extend(similarities)
        all_uuids.extend(uuids_batch)

    all_similarities = np.array(all_similarities)
    top_indices = all_similarities.argsort()[-top_n:][::-1]
    return [all_uuids[i] for i in top_indices]


def load_image_paths_from_db(db_path, uuids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(uuids))
    query = f"SELECT uuid, file_path FROM images WHERE uuid IN ({placeholders})"
    cursor.execute(query, uuids)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def plot_images(main_image_paths, top_similar_images_paths):
    num_input_images = len(main_image_paths)
    num_similar_images = len(top_similar_images_paths)
    num_columns = max(num_input_images, num_similar_images)

    fig, axes = plt.subplots(2, num_columns, figsize=(20, 10))

    for i, img_path in enumerate(main_image_paths):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Input Image {i+1}")
        axes[0, i].axis("off")

    for i, img_path in enumerate(top_similar_images_paths):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Similar Image {i+1}")
        axes[1, i].axis("off")

    # Hide any unused subplots
    for j in range(len(main_image_paths), num_columns):
        axes[0, j].axis("off")
    for j in range(len(top_similar_images_paths), num_columns):
        axes[1, j].axis("off")

    plt.suptitle("Top Similar Images")
    plt.show()


# Example usage
input_image_paths = ["new_images/eye.jpg"]  # Paths to the input images
pca_embeddings_path = "pca_embeddings.pkl"  # pca_embeddings_image_analysis.pkl  (Besser)    # Path to PCA-reduced dataset embeddings
pca_model_path = "pca_model.pkl"  # pca_model_image_analysis.pkl  (Besser)  # Path to the saved PCA model
database_path = "image_metadata.db"  # Path to the database
batch_size = 1000  # Adjust batch size according to memory capacity

# Load PCA model
with open(pca_model_path, "rb") as f:
    pca = pickle.load(f)

# Compute embedding for the new image(s)
if len(input_image_paths) > 1:
    print(f"Computing embeddings for the new images: {input_image_paths}")
    new_image_embedding = compute_multiple_embeddings(input_image_paths, model)
    print("Embeddings computed.")
else:
    print(f"Computing embedding for the new image: {input_image_paths[0]}")
    new_image_embedding = compute_embedding(input_image_paths[0], model)
    print("Embedding computed.")

# Load embeddings in batches and find top similar images
print("Finding top similar images...")
embeddings_batches = load_embeddings_in_batches(pca_embeddings_path, batch_size)
top_similar_images = find_top_similar_images(
    new_image_embedding, embeddings_batches, pca
)
print(f"Top similar images: {top_similar_images}")

# Load image paths from the database
print("Loading image paths from the database...")
image_paths_dict = load_image_paths_from_db(database_path, top_similar_images)
print(f"Image paths loaded: {image_paths_dict}")

# Get the paths for the similar images
top_similar_images_paths = [image_paths_dict[uuid] for uuid in top_similar_images]
print(f"Top similar images paths: {top_similar_images_paths}")

# Plot the images
print("Plotting images...")
plot_images(input_image_paths, top_similar_images_paths)
print("Images plotted.")
