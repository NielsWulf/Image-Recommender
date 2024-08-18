test

# Image-Recommender
A Python-based image recommendation system based on a large image database. 
This project uses various similarity metrics and dimensionality reduction techniques 
to find the best matches for one or more input images.

## Requirments

Python 3.x

NumPy

pandas

scikit-learn

OpenCV

Pillow

matplotlib

Plotly

tqdm

SQLite

PyTorch (for Deep Learning Embeddings)

UMAP

## Usage


### 1) Setting up Database [meta_data.py]

This script processes a directory of image files, extracts metadata (such as file name, path, image dimensions, etc.) 
and stores this information in a SQLite database. 
It is designed to process large data sets efficiently by processing images in batches, 
logging errors and supporting checkpoints to continue processing in case of interruptions.

Setup Database: The script will automatically set up the SQLite database and create the necessary table when run.

Configure Paths: Set the following paths in the script:

    root_directory: The root directory containing the images.
    database_path: The path where the SQLite database will be saved.
    checkpoint_path: The path for saving the checkpoint file.
    log_file: The path for saving the log file.

Run the Script: Execute the script to begin processing images:

```bash```
python script_name.py


The script will iterate through the image files in the specified directory,
extract metadata, and insert it into the SQLite database.


Notes

- The script skips files that cannot be read as images.
- Errors during processing are logged to the specified log file.
- The database is reset (table images is dropped and recreated) each time the script is run.

### 2) Creating color histograms [similarities_color_histograms.py]

This Python script processes a collection of images stored in a SQLite database, generates color histograms for each image, and saves the resulting data into a pickle file for further analysis.

- **Image Preprocessing**: Reads and preprocesses images from the provided file paths.
- **Color Histogram Calculation**: Computes a color histogram for each image using OpenCV.
- **Batch Processing**: Efficiently processes images in batches to handle large datasets.
- **Checkpointing**: Supports resuming from the last processed image using checkpoint files.
- **Pickle Data Storage**: Stores the generated color histograms in a pickle file for easy retrieval.

Run the Script: Execute the script to generate color histograms for your images.

```bash```
python generate_color_histograms.py

Checkpoints: The script uses checkpoints to resume from the last processed image in case of interruption. The checkpoints are saved in a file named checkpoint_histograms.pkl.

preprocess_image(image_path: str) -> np.ndarray:

Reads and preprocesses an image from the provided file path.

calculate_histogram(image: np.ndarray) -> np.ndarray:

calculates a color histogram for the given image using 16 bins for each color channel (RGB).

generate_color_histograms(db_path: str, batch_size: int, combined_pickle_path: str, checkpoint_path: str):

Connects to the SQLite database, retrieves image paths, and generates color histograms in batches. The results are saved in a combined_color_histograms.pkl file, and progress is tracked using checkpoints.

The batchsize might be adjusted from 500

### 3) Creating Image Embeddings [similarities_embeddings.py]

This Python script processes a collection of images stored in a SQLite database, generates embeddings for each image using a pre-trained ResNet model, and saves the resulting data into a pickle file for further analysis.

- **Image Preprocessing**: Reads and preprocesses images from the provided file paths, including resizing and normalizing them for input into a ResNet model.
- **Embedding Generation**: Uses a pre-trained ResNet-50 model (from torchvision) to extract embeddings for each image.
- **Batch Processing**: Efficiently processes images in batches to handle large datasets.
- **Checkpointing**: Supports resuming from the last processed image using checkpoint files.
- **Pickle Data Storage**: Stores the generated embeddings in a pickle file for easy retrieval.

Run the Script: Execute the script to generate embeddings for your images.

```bash```
python generate_embeddings.py

Checkpoints: The script uses checkpoints to resume from the last processed image in case of interruption. The checkpoints are saved in a file named checkpoint_embeddings.pkl.

Function Details:

preprocess_image(image, preprocess, device) -> torch.Tensor:
Reads and preprocesses an image from the provided file path, 
applying the necessary transformations to make it compatible with the ResNet model.

generate_embeddings(db_path: str, batch_size: int, combined_pickle_path: str, checkpoint_path: str):
Connects to the SQLite database, retrieves image paths, and generates embeddings in batches using the ResNet-50 model. The results are saved in a combined_embeddings.pkl file, and progress is tracked using checkpoints.


### 4) Dimensionality Reduction with Incremental and Kernel PCA [pca_similarity.py]

```bash
python dimensionality_reduction.py
```

pathes and parameters: 


    embeddings_pickle_path: Path to the pickle file containing the original embeddings.
    
    pca_pickle_path: Path where the Incremental PCA reduced embeddings will be saved.
    
    pca_model_path: Path to save the trained Incremental PCA model.
    
    kernel_pca_pickle_path: Path to save the Kernel PCA reduced embeddings.
    
    n_components_ipca: Number of components for Incremental PCA (default is 50).
    
    n_components_kpca: Number of components for Kernel PCA (default is 100).
    
    batch_size: Batch size for Incremental PCA (default is 10,000).
    
    kernel_pca_sample_size: Number of samples to use for Kernel PCA (default is 10,000).

This Python script performs dimensionality reduction on a set of image embeddings using both Incremental PCA and Kernel PCA. The reduced embeddings are then saved to pickle files for further analysis.

- **Incremental PCA**: Reduces the dimensionality of large datasets using Incremental PCA, which processes the data in batches, making it suitable for handling large-scale data.
- **Kernel PCA**: Further reduces the dimensionality of the embeddings using Kernel PCA on a subsampled dataset to capture non-linear relationships.
- **Batch Processing**: Efficiently handles large datasets by processing data in batches during Incremental PCA.
- **Subsampling for Kernel PCA**: Allows for the selection of a random subset of the data to make Kernel PCA computationally feasible.
- **Pickle Data Storage**: Stores the reduced embeddings and trained models in pickle files for easy retrieval and further use.


### 5) Similarity Measures

#### similarity_Measure_Color_Profile.py : 

```bash
similarity_Measure_Color_Profile.py
```

This Python script compares the color histograms of a target image (or a set of images) against a large collection of precomputed histograms stored in a pickle file. It identifies and displays the most visually similar images based on the selected comparison method.


Comparison Method: You can change the method parameter in compare_histograms to use different similarity metrics (e.g., correlation, chi-square, intersection, and Bhattacharyya).
We reccomend using correlation or Bhattacharyya



## Results from correlation


![grafik](https://github.com/user-attachments/assets/0cc2ae16-7ac9-4118-9721-e9d25cb55605)


## Results from Bhattacharyya


![grafik](https://github.com/user-attachments/assets/82a0aa8d-473e-4076-abdc-29c17ca8c8d4)


#### similarity_Measure_Embeddings.py: 

```bash
similarity_Measure_Embeddings.py
```

This Python script uses a pre-trained ResNet model to generate image embeddings for the input and compares them to a database of precomputed embeddings to find the most visually similar images. 

Comparison Method: The similarity is measured using cosine similarity, which is ideal for comparing high-dimensional vectors like image embeddings.

Top N Results: Modify the top_n parameter to control how many of the most similar images are returned and displayed.

## Resluts from Embeddings

![grafik](https://github.com/user-attachments/assets/0ab82704-fb16-4b74-866b-ff1e8d5c8f0c)





#### similarity_Measure_PCA_Embeddings.py:

```bash
similarity_Measure_PCA_Embeddings.py
```

Execute the script to compute the embedding of an input image, apply PCA, and find the most similar images in the dataset that was also dimension reduced.




## Results for PCA Embeddings: 


![grafik](https://github.com/user-attachments/assets/19f24fc5-2478-4101-8b63-969c08cfa3d7)
