import pandas as pd
import pytest
from pathlib import Path
from unittest import mock
import sys
import torch

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from embedding_ag_news.do_embeddings import create_embeddings, mean_pooling
from transformers import AutoTokenizer, AutoModel


def test_mean_pooling():
    """
    Test the mean_pooling function to ensure it handles the tensor operations correctly.
    """
    # Create mock data for testing
    model_output = (
        torch.randn(1, 3, 768),
    )  # Mocking a model output of shape (batch_size, sequence_length, hidden_size)
    attention_mask = torch.tensor([[1, 1, 0]])  # Mocking an attention mask

    # Perform mean pooling
    pooled_output = mean_pooling(model_output, attention_mask)

    # Check output shape and values
    assert pooled_output.shape == (
        1,
        768,
    ), "Output shape is inconsistent with expected shape."
    assert not torch.isnan(pooled_output).any(), "Output contains NaN values."


def test_create_embeddings():
    """
    Test the create_embeddings function to ensure it returns the correct output format.
    """
    # mock dataframe
    data = {"text": ["This is a test sentence.", "This is another test sentence."]}
    df = pd.DataFrame(data)

    # Load a real tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = create_embeddings(df, tokenizer, model)

    # "Maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search." See: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    assert isinstance(embeddings, list), "Embeddings should be a list."
    assert all(
        isinstance(e, list) for e in embeddings
    ), "Each embedding should be a list."
    assert all(
        len(e) == 384 for e in embeddings
    ), "Each embedding should have 768 dimensions."


# This part is necessary to run the tests
if __name__ == "__main__":
    pytest.main([__file__])
