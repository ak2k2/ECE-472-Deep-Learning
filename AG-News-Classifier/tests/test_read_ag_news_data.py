import pandas as pd
import pytest
from pathlib import Path
from unittest import mock
import sys

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

# Adjust the import based on your package structure
from load_ag_news_dataset.read_ag_news_data import load_ag_dataframes


def test_load_ag_news_data_as_df():
    """
    Test the read_ag_news_data function to ensure it correctly reads the
    AG News data from pickle files and returns DataFrames.
    """
    # Mock the pickle.load function to return a sample DataFrame
    # This avoids the need to read an actual file from disk
    with mock.patch("pickle.load") as mock_pickle_load:
        mock_pickle_load.return_value = pd.DataFrame(
            {"text": ["sample text"], "label": [1]}
        )

        # Call the function
        test_df, train_df = load_ag_dataframes()

        # Verify the returned value is a DataFrame with the expected columns
        assert isinstance(test_df, pd.DataFrame)
        assert "text" in test_df.columns
        assert "label" in test_df.columns

        assert isinstance(train_df, pd.DataFrame)
        assert "text" in train_df.columns
        assert "label" in train_df.columns

        # Verify the DataFrame is not empty
        assert not test_df.empty
        assert not train_df.empty


# This part is necessary to run the test
if __name__ == "__main__":
    pytest.main([__file__])
