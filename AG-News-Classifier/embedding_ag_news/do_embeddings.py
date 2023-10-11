import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from load_ag_news_dataset.read_ag_news_data import load_ag_dataframes
import pandas as pd


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def create_embeddings(dataframe, tokenizer, model):
    # Convert the text data to embeddings
    sentences = dataframe["text"].tolist()
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt", max_length=512
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Convert tensor of embeddings to list of embeddings
    embeddings = sentence_embeddings.tolist()
    return embeddings


def main():
    # Load the data
    train_df, test_df = load_ag_dataframes()

    # take a subset of the data
    train_df = train_df[:100]
    test_df = test_df[:100]

    # Load pre-trained model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Create embeddings for train and test set
    train_df["embeddings"] = create_embeddings(train_df, tokenizer, model)
    test_df["embeddings"] = create_embeddings(test_df, tokenizer, model)

    # train_df and test_df have a new column 'embeddings' that contain the sentence embeddings.
    train_df.to_pickle("embedding-ag-news/embedded-datasets/train_with_embeddings.pkl")
    test_df.to_pickle("embedding-ag-news/embedded-datasets/test_with_embeddings.pkl")

    print("Embeddings are successfully created and dataframes are saved.")


if __name__ == "__main__":
    main()
