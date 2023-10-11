import pandas as pd
import pickle
from datasets import load_dataset
import os


def get_ag_news() -> tuple[pd.DataFrame]:
    dataset = load_dataset("ag_news")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, test_df


def save_as_pickle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    directory: str = "AG-News-Classifier/AG-News-Data",
):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    train_path = os.path.join(directory, "train_df.pkl")
    test_path = os.path.join(directory, "test_df.pkl")

    # Save dataframes
    with open(train_path, "wb") as f:
        pickle.dump(train_df, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_df, f)

    print(f"Dataframes saved in '{directory}' directory")


if __name__ == "__main__":
    train_df, test_df = get_ag_news()
    save_as_pickle(train_df, test_df)
