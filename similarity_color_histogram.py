import os
import gc
import sqlite3
import pickle
from typing import List, Dict
from tqdm import tqdm
import cv2
import numpy as np

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

def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at path {image_path}")
    return image

def calculate_histogram(image: np.ndarray) -> np.ndarray:
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    return color_hist.flatten()

def generate_color_histograms(db_path: str, batch_size: int, combined_pickle_path: str, checkpoint_path: str):
    print("Generating color histograms...")
    
    processed_uuids = load_checkpoint(checkpoint_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uuid, file_path FROM images")
    rows = cursor.fetchall()

    total_images = len(rows)
    progress_bar = tqdm(total=total_images, desc="Processing Images", unit="image")

    combined_data = {}
    
    try:
        for start_idx in range(0, total_images, batch_size):
            batch = rows[start_idx:start_idx + batch_size]
            metadata_batch = []

            for image_uuid, file_path in batch:
                if image_uuid in processed_uuids:
                    continue
                try:
                    image = preprocess_image(file_path)
                    color_hist = calculate_histogram(image)

                    metadata_batch.append({
                        'uuid': image_uuid,
                        'color_histogram': color_hist
                    })
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")

            if metadata_batch:
                combined_data.update({meta['uuid']: meta['color_histogram'] for meta in metadata_batch})
                processed_uuids.update([meta['uuid'] for meta in metadata_batch])
                save_checkpoint(checkpoint_path, processed_uuids)

                progress_bar.update(len(metadata_batch))
                del metadata_batch
                gc.collect()

    except Exception as e:
        print(f"An error occurred: {e}")

    save_to_pickle(combined_data, combined_pickle_path)
    progress_bar.close()
    conn.close()

if __name__ == "__main__":
    db_path = "image_metadata.db"
    batch_size = 500
    combined_pickle_path = "combined_color_histograms.pkl"
    checkpoint_path = "checkpoint_histograms.pkl"
    
    generate_color_histograms(db_path, batch_size, combined_pickle_path, checkpoint_path)
