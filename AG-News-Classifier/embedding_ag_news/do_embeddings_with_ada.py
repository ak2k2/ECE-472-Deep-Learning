import concurrent.futures
import logging
import sys
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from load_ag_news_dataset.read_ag_news_data import load_ag_dataframes

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get the embedding for a text using openai.Embedding.create().
    """
    text = text.replace("\n", " ")  # Ensure no newlines
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]


def process_and_save_embeddings(df, output_path):
    """
    Process the embeddings in parallel using concurrent.futures.ThreadPoolExecutor.
    """
    texts = df["text"].tolist()
    # ThreadPool to make requests in parallel (avoid rate limit by keeping the number of requests per minute low)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Use tqdm
        embeddings = list(tqdm(executor.map(get_embedding, texts), total=len(texts)))

    # Add embeddings col the DataFrame
    df["embedding"] = embeddings
    df.to_pickle(output_path)


def main():
    test_df, train_df = load_ag_dataframes()
    train_df = train_df[:50000]
    process_and_save_embeddings(
        train_df, "embedding_ag_news/embedded-datasets/train_with_embeddings.pkl"
    )
    process_and_save_embeddings(
        test_df, "embedding_ag_news/embedded-datasets/test_with_embeddings.pkl"
    )


if __name__ == "__main__":
    main()
