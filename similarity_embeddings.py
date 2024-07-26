import os
import gc
import sqlite3
import pickle
import uuid
from typing import List, Dict
from tqdm import tqdm
import cv2
import torch
import numpy as np
from torchvision import models, transforms

def save_to_pickle(data: Dict, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(file_path: str) -> Dict:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return {}

def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    return set()

def save_checkpoint(checkpoint_path: str, processed_uuids: set):
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(processed_uuids, f)

def preprocess_image(image, preprocess, device):
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    return image_preprocessed

def generate_embeddings(db_path: str, batch_size: int, checkpoint_path: str):
    print("Generating embeddings...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    model = torch.nn.Sequential(*list(model.children())[:-1])

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    processed_uuids = load_checkpoint(checkpoint_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uuid, file_path FROM images")
    rows = cursor.fetchall()

    total_images = len(rows)
    progress_bar = tqdm(total=min(total_images, 150000), desc="Processing Images", unit="image")

    def process_batch(batch):
        metadata_batch = []
        for image_uuid, file_path in batch:
            if image_uuid in processed_uuids:
                continue
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Unable to read image at path {file_path}")

                image_preprocessed = preprocess_image(image, preprocess, device)

                with torch.no_grad():
                    features = model(image_preprocessed)
                    embedding = features.view(features.size(0), -1).cpu().numpy().flatten()

                metadata_batch.append({
                    'uuid': image_uuid,
                    'embedding': embedding
                })
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")

        return metadata_batch

    try:
        for start_idx in range(0, min(total_images, 150000), batch_size):
            batch = rows[start_idx:start_idx + batch_size]
            metadata_batch = process_batch(batch)

            if metadata_batch:
                temp_file_name = f"embeddings_{uuid.uuid4()}.pkl"
                save_to_pickle({meta['uuid']: meta['embedding'] for meta in metadata_batch}, temp_file_name)

                processed_uuids.update([meta['uuid'] for meta in metadata_batch])
                save_checkpoint(checkpoint_path, processed_uuids)

                progress_bar.update(len(metadata_batch))

                del metadata_batch
                gc.collect()

    except Exception as e:
        print(f"An error occurred: {e}")

    progress_bar.close()
    conn.close()

if __name__ == "__main__":
    db_path = "image_metadata.db"
    batch_size = 500
    checkpoint_path = "checkpoint_embeddings.pkl"
    
    generate_embeddings(db_path, batch_size, checkpoint_path)
