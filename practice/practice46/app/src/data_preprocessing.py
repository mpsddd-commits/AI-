# app/src/data_preprocessing.py

import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Example preprocessing steps.
    """
    # Fill missing values
    df = df.fillna(0)
    return df
