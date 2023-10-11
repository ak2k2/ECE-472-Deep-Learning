import pandas as pd
import pytest
from pathlib import Path
from unittest import mock
import sys
import torch

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from embedding_ag_news.get_embedded_data import read_ag_news_data
from transformers import AutoTokenizer, AutoModel


# Path to your embedded dataset
EMBEDDED_DATASET_PATH = "embedding_ag_news/embedded-datasets/test_with_embeddings.pkl"


def test_read_ag_news_data():
    # Load the dataset
    embedded_dataset = pd.read_pickle(EMBEDDED_DATASET_PATH)

    # Check the structure of the loaded dataset
    assert isinstance(embedded_dataset, pd.DataFrame)
    assert "text" in embedded_dataset.columns
    assert "label" in embedded_dataset.columns
    assert "embeddings" in embedded_dataset.columns

    # Check the type of the embeddings column values
    assert isinstance(embedded_dataset.iloc[0]["embeddings"], list)

    # Confirm embeddings are of type float
    assert all(isinstance(val, float) for val in embedded_dataset.iloc[0]["embeddings"])

    # Correct length
    assert len(embedded_dataset.iloc[0]["embeddings"]) == 384


if __name__ == "__main__":
    pytest.main([__file__])
