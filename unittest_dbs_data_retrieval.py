import pytest
from unittest.mock import patch, MagicMock
import sqlite3
from similarity_Measure_Color_Profile import load_image_paths_from_db


def test_load_image_paths_from_db_valid_uuids():
    with patch("sqlite3.connect") as mock_connect:
        # Setup mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock the database rows
        sample_uuids = ["uuid1", "uuid2", "uuid3"]
        sample_paths = [
            "path/to/image1.jpg",
            "path/to/image2.jpg",
            "path/to/image3.jpg",
        ]
        mock_cursor.fetchall.return_value = list(zip(sample_uuids, sample_paths))

        # Call the function
        result = load_image_paths_from_db("fake_db_path.db", sample_uuids)

        # Verify the output
        expected_result = dict(zip(sample_uuids, sample_paths))
        assert result == expected_result

        # Ensure the correct SQL query was executed
        mock_cursor.execute.assert_called_once()
        assert (
            "SELECT uuid, file_path FROM images WHERE uuid IN"
            in mock_cursor.execute.call_args[0][0]
        )


def test_load_image_paths_from_db_missing_uuids():
    with patch("sqlite3.connect") as mock_connect:
        # Setup mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock the database rows with only one UUID found
        sample_uuids = ["uuid1", "uuid2", "uuid3"]
        found_uuids = ["uuid1"]
        found_paths = ["path/to/image1.jpg"]
        mock_cursor.fetchall.return_value = list(zip(found_uuids, found_paths))

        # Call the function
        result = load_image_paths_from_db("fake_db_path.db", sample_uuids)

        # Verify the output
        expected_result = dict(zip(found_uuids, found_paths))
        assert result == expected_result

        # Ensure the correct SQL query was executed
        mock_cursor.execute.assert_called_once()
        assert (
            "SELECT uuid, file_path FROM images WHERE uuid IN"
            in mock_cursor.execute.call_args[0][0]
        )


def test_load_image_paths_from_db_connection_failure():
    with patch("sqlite3.connect") as mock_connect:
        # Simulate a database connection failure
        mock_connect.side_effect = sqlite3.OperationalError(
            "Unable to connect to the database"
        )

        with pytest.raises(sqlite3.OperationalError):
            load_image_paths_from_db("fake_db_path.db", ["uuid1", "uuid2"])


def test_load_image_paths_from_db_empty_result():
    with patch("sqlite3.connect") as mock_connect:
        # Setup mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock the database to return no rows
        mock_cursor.fetchall.return_value = []

        # Call the function
        result = load_image_paths_from_db("fake_db_path.db", ["uuid1", "uuid2"])

        # Verify the output is an empty dictionary
        assert result == {}

        # Ensure the correct SQL query was executed
        mock_cursor.execute.assert_called_once()
        assert (
            "SELECT uuid, file_path FROM images WHERE uuid IN"
            in mock_cursor.execute.call_args[0][0]
        )
