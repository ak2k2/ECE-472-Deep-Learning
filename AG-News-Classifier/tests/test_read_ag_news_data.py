import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from load_ag_news_dataset.read_ag_news_data import load_ag_dataframes


def test_load_ag_news_data_as_df():
    """
    Test the read_ag_news_data function to ensure it correctly reads the
    AG News data from pickle files and returns DataFrames.
    """
    # Mock the pickle.load function to return a sample DataFrame
    with mock.patch("pickle.load") as mock_pickle_load:
        mock_pickle_load.return_value = pd.DataFrame(
            {"text": ["sample text"], "label": [1]}
        )

        # Call the function
        train_df, test_df = load_ag_dataframes()

        assert train_df.shape == (1, 2)
        assert test_df.shape == (1, 2)

        # Verify the returned value is a DataFrame with the expected columns
        assert isinstance(test_df, pd.DataFrame)
        assert "text" in test_df.columns
        assert "label" in test_df.columns

        assert isinstance(train_df, pd.DataFrame)
        assert "text" in train_df.columns
        assert "label" in train_df.columns

        assert not test_df.empty
        assert not train_df.empty


if __name__ == "__main__":
    pytest.main([__file__])
