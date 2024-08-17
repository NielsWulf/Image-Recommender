import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from similarity_Measure_PCA_Embeddings import compute_embedding
import torch


class TestPCAComputeEmbedding(unittest.TestCase):

    @patch("similarity_Measure_PCA_Embeddings.Image.open")
    @patch("similarity_Measure_PCA_Embeddings.preprocess")
    @patch("similarity_Measure_PCA_Embeddings.model")
    def test_compute_embedding_valid_image(
        self, mock_model, mock_preprocess, mock_image_open
    ):
        # Setup mock for image
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        # Setup mock for preprocessing
        mock_tensor = MagicMock()
        mock_preprocess.return_value = mock_tensor

        # Setup mock for model
        mock_output = torch.randn(
            1, 2048
        )  # Expected output size from ResNet50 without the last layer
        mock_model.return_value = mock_output

        # Call the function
        embedding = compute_embedding("dummy_image_path.jpg", mock_model)

        # Check if the output is as expected
        self.assertEqual(embedding.shape, (2048,))
        self.assertIsInstance(embedding, np.ndarray)

    @patch("similarity_Measure_PCA_Embeddings.Image.open")
    def test_compute_embedding_invalid_image(self, mock_image_open):
        # Setup mock to raise an exception for invalid image
        mock_image_open.side_effect = IOError("Unable to open image")

        with self.assertRaises(IOError):
            compute_embedding("invalid_image_path.jpg", MagicMock())

    @patch("similarity_Measure_PCA_Embeddings.Image.open")
    @patch("similarity_Measure_PCA_Embeddings.preprocess")
    @patch("similarity_Measure_PCA_Embeddings.model")
    def test_compute_embedding_consistency(
        self, mock_model, mock_preprocess, mock_image_open
    ):
        # Setup mocks
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_tensor = MagicMock()
        mock_preprocess.return_value = mock_tensor
        mock_output = torch.randn(1, 2048)
        mock_model.return_value = mock_output

        # Call the function multiple times
        embedding1 = compute_embedding("dummy_image_path.jpg", mock_model)
        embedding2 = compute_embedding("dummy_image_path.jpg", mock_model)

        # Check if the output is consistent
        np.testing.assert_array_equal(embedding1, embedding2)


if __name__ == "__main__":
    unittest.main()
