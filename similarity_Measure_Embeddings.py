import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained ResNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last layer
model = model.to(device)  # Move the model to the device (GPU or CPU)
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
        data = pd.read_pickle(f)
    uuids = list(data.keys())
    embeddings = list(data.values())
    for i in range(0, len(embeddings), batch_size):
        yield uuids[i : i + batch_size], embeddings[i : i + batch_size]


def compute_embedding(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(
        device
    )  # Move the tensor to the same device as the model
    with torch.no_grad():
        embedding = model(img_tensor)
    embedding = (
        embedding.view(-1).cpu().numpy()
    )  # Flatten the embedding and move to CPU
    return embedding


def compute_multiple_embeddings(img_paths, model):
    batch_tensors = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img)
        batch_tensors.append(img_tensor)

    batch_tensor = torch.stack(batch_tensors).to(
        device
    )  # Create a batch tensor and move to the device
    with torch.no_grad():
        embeddings = model(batch_tensor)

    embeddings = (
        embeddings.view(len(img_paths), -1).cpu().numpy()
    )  # Flatten and move to CPU
    return np.mean(embeddings, axis=0)  # Average the embeddings


def find_top_similar_images(embedding, embeddings_batches, top_n=5):
    all_similarities = []
    all_uuids = []

    for uuids_batch, embeddings_batch in embeddings_batches:
        embeddings_batch = np.array(embeddings_batch)
        similarities = cosine_similarity([embedding], embeddings_batch)[0]
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
    plt.figure(figsize=(20, 10))

    for i, img_path in enumerate(main_image_paths):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Input image not found at path: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Input Image {i+1}")
        plt.axis("off")

    for i, img_path in enumerate(top_similar_images_paths):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Similar image not found at path: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 5, i + 6)
        plt.imshow(img)
        plt.title(f"Similar Image {i+1}")
        plt.axis("off")

    plt.show()


# Example usage
input_images = [
    "new_images/new_try2.jpg",
    "new_images/turtle.jpg",
]  # Paths to the input images
cosine_similarities_path = (
    "combined_embeddings.pkl"  # Path to precomputed dataset embeddings
)
database_path = "image_metadata.db"  # Path to the database
batch_size = 1000  # Adjust batch size according to memory capacity
multiple_inputs = len(input_images) > 1

# Compute embedding for the new image(s)
if multiple_inputs:
    print(f"Computing embeddings for the new images: {input_images}")
    new_image_embedding = compute_multiple_embeddings(input_images, model)
    print("Embeddings computed.")
else:
    print(f"Computing embedding for the new image: {input_images[0]}")
    new_image_embedding = compute_embedding(input_images[0], model)
    print("Embedding computed.")

# Load embeddings in batches and find top similar images
print("Finding top similar images...")
embeddings_batches = load_embeddings_in_batches(cosine_similarities_path, batch_size)
top_similar_images = find_top_similar_images(new_image_embedding, embeddings_batches)
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
plot_images(input_images, top_similar_images_paths)
print("Images plotted.")
