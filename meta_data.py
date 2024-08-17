import os
import sqlite3
import uuid
from typing import List, Dict, Generator
from tqdm import tqdm
import cv2
import pickle


# Data Generator
def find_image_files_with_metadata(
    root_dir: str, batch_size: int = 500, log_file: str = "processing_log.txt"
) -> Generator[List[Dict], None, None]:
    metadata_batch = []
    with open(log_file, "w") as log:
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(subdir, file)
                    image = cv2.imread(full_path)
                    if image is not None:
                        try:
                            height, width, _ = image.shape

                            unique_id = str(uuid.uuid4())
                            metadata_batch.append(
                                {
                                    "uuid": unique_id,
                                    "file_name": file,
                                    "file_path": full_path,
                                    "directory": subdir,
                                    "width": width,
                                    "height": height,
                                }
                            )
                        except Exception as e:
                            log.write(f"Error processing file {full_path}: {e}\n")
                    else:
                        log.write(f"Skipping file {full_path}: Unable to read image\n")

                    if len(metadata_batch) == batch_size:
                        yield metadata_batch
                        metadata_batch = []

        if metadata_batch:
            yield metadata_batch


# Database Setup
def setup_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS images")
    cursor.execute(
        """
    CREATE TABLE images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
        uuid TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        directory TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL
    )"""
    )
    conn.commit()
    conn.close()


def insert_metadata_batch(conn: sqlite3.Connection, metadata_batch: List[Dict]):
    cursor = conn.cursor()
    cursor.executemany(
        """
    INSERT INTO images (uuid, file_name, file_path, directory, width, height)
    VALUES (:uuid, :file_name, :file_path, :directory, :width, :height)
    """,
        metadata_batch,
    )
    conn.commit()


def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return set()


def save_checkpoint(checkpoint_path: str, processed_uuids: set):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(processed_uuids, f)


# Example usage
if __name__ == "__main__":
    root_directory = "E:/data/image_data"
    database_path = "image_metadata.db"
    checkpoint_path = "checkpoint.pkl"
    log_file = "processing_log.txt"

    setup_database(database_path)

    conn = sqlite3.connect(database_path)

    total_images = sum([len(files) for r, d, files in os.walk(root_directory) if files])
    progress_bar = tqdm(total=total_images, desc="Processing Images", unit="image")

    processed_uuids = load_checkpoint(checkpoint_path)
    metadata_generator = find_image_files_with_metadata(
        root_directory, log_file=log_file
    )

    try:
        for metadata_batch in metadata_generator:
            # Filter out already processed images
            metadata_batch = [
                meta for meta in metadata_batch if meta["uuid"] not in processed_uuids
            ]

            if metadata_batch:
                insert_metadata_batch(conn, metadata_batch)
                processed_uuids.update([meta["uuid"] for meta in metadata_batch])
                progress_bar.update(len(metadata_batch))

                # Save checkpoint after each batch
                save_checkpoint(checkpoint_path, processed_uuids)

    except Exception as e:
        with open(log_file, "a") as log:
            log.write(f"An error occurred: {e}\n")

    progress_bar.close()

    conn.close()
