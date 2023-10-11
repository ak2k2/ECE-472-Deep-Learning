import pandas as pd
from pathlib import Path
import sys

script_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(script_dir))

from load_ag_news_dataset.read_ag_news_data import (
    load_data,
)  # reuse the load_data function from read_ag_news_data.py


def read_ag_news_data():
    """
    Read AG News data from pickle files.
    :return: tuple of DataFrames, (test_df, train_df)
    """
    # Load the data into pandas DataFrames
    test_pickle_path = "embedding_ag_news/embedded-datasets/test_with_embeddings.pkl"
    train_pickle_path = "embedding_ag_news/embedded-datasets/test_with_embeddings.pkl"
    test_df = load_data(test_pickle_path)
    train_df = load_data(train_pickle_path)

    return test_df, train_df


df1, df2 = read_ag_news_data()

print(df1.head())
