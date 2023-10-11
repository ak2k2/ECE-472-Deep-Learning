import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from embedding_ag_news.get_embedded_data import read_ag_news_data
from transformers import AutoModel, AutoTokenizer

# Path to your embedded dataset
EMBEDDED_DATASET_PATH = "embedding_ag_news/embedded-datasets/test_with_embeddings.pkl"


# def test_read_ag_news_data_miniLM():
#     # Load the dataset
#     embedded_dataset = pd.read_pickle(EMBEDDED_DATASET_PATH)

#     # Check the structure of the loaded dataset
#     assert isinstance(embedded_dataset, pd.DataFrame)
#     assert "text" in embedded_dataset.columns
#     assert "label" in embedded_dataset.columns
#     assert "embeddings" in embedded_dataset.columns

#     # Check the type of the embeddings column values
#     assert isinstance(embedded_dataset.iloc[0]["embeddings"], list)

#     # Confirm embeddings are of type float
#     assert all(isinstance(val, float) for val in embedded_dataset.iloc[0]["embeddings"])

#     # Correct length
#     assert len(embedded_dataset.iloc[0]["embeddings"]) == 384


def test_read_ag_news_data_ada():
    # Load the dataset
    embedded_dataset = pd.read_pickle(EMBEDDED_DATASET_PATH)

    # Check the structure of the loaded dataset
    assert isinstance(embedded_dataset, pd.DataFrame)
    assert "text" in embedded_dataset.columns
    assert "label" in embedded_dataset.columns
    assert "embedding" in embedded_dataset.columns

    # Check the type of the embeddings column values
    assert isinstance(embedded_dataset.iloc[0]["embedding"], list)

    # Confirm embeddings are of type float
    assert all(isinstance(val, float) for val in embedded_dataset.iloc[0]["embedding"])

    # Correct length
    assert len(embedded_dataset.iloc[0]["embedding"]) == 1536


if __name__ == "__main__":
    pytest.main([__file__])
