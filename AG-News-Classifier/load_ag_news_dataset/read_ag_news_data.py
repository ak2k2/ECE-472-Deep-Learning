import pandas as pd
import pickle


def load_data(pickle_file_path):
    """
    Load DataFrame from pickle file.
    :param pickle_file_path: str, path to the pickle file
    :return: DataFrame, the loaded data
    """
    try:
        with open(pickle_file_path, "rb") as file:
            df = pickle.load(file)
        return df
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None


def load_ag_dataframes():
    """
    Read AG News data from pickle files.
    :return: tuple of DataFrames, (test_df, train_df)
    """
    # Define the paths to your pickle files
    test_pickle_path = "AG-News-Data/test_df.pkl"
    train_pickle_path = "AG-News-Data/train_df.pkl"

    # Load the data into pandas DataFrames
    test_df = load_data(test_pickle_path)
    train_df = load_data(train_pickle_path)

    return test_df, train_df
