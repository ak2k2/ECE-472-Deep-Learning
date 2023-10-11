import pandas as pd
import pickle
from datasets import load_dataset
import os


def get_ag_news() -> tuple:
    """
    Downloads the AG News dataset and converts it into pandas DataFrames.

    Returns:
        tuple: A tuple containing two DataFrames, the training and the testing set.
    """
    dataset = load_dataset("ag_news")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    return train_df, test_df


def save_as_pickle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    directory: str = "AG-News-Data",
):
    """
    Saves DataFrames as pickle files in the specified directory.

    Args:
        train_df (pd.DataFrame): The DataFrame containing the training set.
        test_df (pd.DataFrame): The DataFrame containing the testing set.
        directory (str, optional): The directory where to save the pickle files.
                                   Defaults to "AG-News-Data".
    """
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
