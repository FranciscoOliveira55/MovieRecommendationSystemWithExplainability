from operator import truth
from typing import Tuple

import pandas as pd

from configs import FilePaths
from datetime import datetime
import json


def write_log(message: str, show_in_console: bool = True):
    """
    Writes a message in the logs.txt file
    :param message:
    :param show_in_console:
    :return:
    """
    # Get current date
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Build message
    log_message = f"{timestamp} -> {message}\n"
    # Write to logs file
    with open(FilePaths.RUNNING_LOGS_PATH, "a", encoding="utf-8") as file:
        file.write(log_message)
        file.flush()
    # If needed print on console
    if show_in_console:
        print(log_message, flush=True)


def filter_json_serializable(d: dict) -> dict:
    """
    Makes sure every value in the dictionary is serializable
    :param d:
    :return: dict:
    """
    serializable = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, OverflowError):
            serializable[k] = str(v)
    return serializable


def save_df_locally_to_csv(
        df_to_save: pd.DataFrame,
        file_path: str
):
    """
    Saves transformed df locally

    :param df_to_save:
    :param file_path:
    :return:
    """
    # Saves the df
    df_to_save.to_csv(file_path, index=False)


def read_df_data_from_csv(file_path: str, columns_names: Tuple[str, ...] = None) -> pd.DataFrame:
    """
    Reads the movies dataset from file and returns dataframe

    :param file_path:
    :param columns_names:
    :return: pd.DataFrame:
    """
    if columns_names is not None:
        return pd.read_csv(file_path, usecols=columns_names)
    else:
        return pd.read_csv(file_path)
