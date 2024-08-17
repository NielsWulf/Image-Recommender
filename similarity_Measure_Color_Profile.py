import os
import pickle
import sqlite3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# Load the histograms from the pickle file
def load_histograms(pickle_file):
    with open(pickle_file, "rb") as f:
        histograms = pickle.load(f)
    return histograms


# Histogram comparison (without JIT because of cv2 usage)
def compare_histograms(hist1, hist2, method="correlation"):
    if hist1.shape != hist2.shape:
        raise ValueError("Histograms must have the same shape for comparison.")

    if method == "correlation":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    elif method == "chi-square":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method == "intersection":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    elif method == "bhattacharyya":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)


# Preprocess histogram (without JIT because of cv2 usage)
def preprocess_histogram(image, bin_size=(16, 16, 16), ranges=[0, 256, 0, 256, 0, 256]):
    hist = cv2.calcHist([image], [0, 1, 2], None, bin_size, ranges)
    cv2.normalize(hist, hist)
    return hist.flatten()


# Compute the average histogram for multiple input images (without JIT)
def compute_average_histogram(image_paths):
    histograms = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read image at path {img_path}")
        hist = preprocess_histogram(image)
        histograms.append(hist)
    return np.mean(histograms, axis=0)


# Find the top similar images using parallel processing
def find_top_color_similar_images(
    target_hist, histograms, top_n=5, method="correlation", batch_size=500
):
    all_uuids = list(histograms.keys())
    similarities = []

    # Use parallel processing for histogram comparison
    def process_batch(batch_histograms):
        batch_similarities = []
        for hist in batch_histograms:
            try:
                similarity = compare_histograms(target_hist, hist, method)
                batch_similarities.append(similarity)
            except ValueError as e:
                batch_similarities.append(
                    float("-inf") if method in ["correlation"] else float("inf")
                )
        return batch_similarities

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(all_uuids), batch_size):
            batch_uuids = all_uuids[i : i + batch_size]
            batch_histograms = [histograms[uuid] for uuid in batch_uuids]
            futures.append(executor.submit(process_batch, batch_histograms))

        for future in tqdm(futures):
            similarities.extend(future.result())

    if method in ["correlation", "intersection"]:
        top_similar_indices = np.argsort(similarities)[-top_n:]
    else:
        top_similar_indices = np.argsort(similarities)[:top_n]

    top_similar_uuids = [all_uuids[idx] for idx in top_similar_indices]
    return top_similar_uuids


# Plot images
def plot_images(input_image_paths, similar_image_paths, method):
    num_input_images = len(input_image_paths)
    num_similar_images = len(similar_image_paths)
    num_columns = max(num_input_images, num_similar_images)

    fig, axes = plt.subplots(2, num_columns, figsize=(20, 10))

    for i, img_path in enumerate(input_image_paths):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Input Image {i+1}")
        axes[0, i].axis("off")

    for i, img_path in enumerate(similar_image_paths):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Similar Image {i+1}")
        axes[1, i].axis("off")

    for j in range(num_input_images, num_columns):
        axes[0, j].axis("off")
    for j in range(num_similar_images, num_columns):
        axes[1, j].axis("off")

    plt.suptitle(f"Top Similar Images using {method} method")
    plt.show()


# Load image paths from the database
def load_image_paths_from_db(db_path, uuids):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(uuids))
    query = f"SELECT uuid, file_path FROM images WHERE uuid IN ({placeholders})"
    cursor.execute(query, uuids)
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def main():
    pickle_file = "combined_color_histograms.pkl"  # Replace with your actual file path
    input_image_paths = [
        "new_images/texture.jpg",
        "new_images/pengu.jpg",
    ]  # Replace with your actual input image paths
    database_path = "image_metadata.db"

    histograms = load_histograms(pickle_file)

    if len(input_image_paths) > 1:
        print(f"Computing average histogram for the new images: {input_image_paths}")
        target_hist = compute_average_histogram(input_image_paths)
    else:
        print(f"Computing histogram for the new image: {input_image_paths[0]}")
        input_image = cv2.imread(input_image_paths[0])
        if input_image is None:
            raise ValueError(f"Unable to read image at path {input_image_paths[0]}")
        target_hist = preprocess_histogram(input_image)

    methods = ["correlation"]  # , 'bhattacharyya', 'chi-square', 'intersection']
    for method in methods:
        top_similar_uuids = find_top_color_similar_images(
            target_hist, histograms, top_n=5, method=method
        )
        image_paths_dict = load_image_paths_from_db(database_path, top_similar_uuids)
        top_similar_image_paths = [image_paths_dict[uuid] for uuid in top_similar_uuids]
        print(f"Top similar images using {method} method: {top_similar_uuids}")
        plot_images(input_image_paths, top_similar_image_paths, method)


if __name__ == "__main__":
    main()
