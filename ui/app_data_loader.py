from typing import Tuple

import numpy as np
import pandas as pd

import configs
from core import data_loader
import streamlit as st
from core.utils import write_log
from configs import UserPickConfig, FilteringType, FilePaths

if __name__ == '__main__':
    pass


def initialize_dataframes(
        ratings_user_id: int,
        clean_and_index_dfs: bool,
        filtering_type: FilteringType
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the full rated and unrated dataframes into Streamlit session state, if not already loaded.

    :param ratings_user_id:
    :param clean_and_index_dfs:
    :param filtering_type:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Gets the rated and unrated dataframes
    rated_df_path, unrated_df_path = FilePaths.get_transformed_df_path(
        selected_user_id=ratings_user_id,
        clean_and_index_dfs=clean_and_index_dfs,
        filtering_type=filtering_type
    )
    if rated_df_path not in st.session_state or True:
        full_rated_df, full_unrated_df = data_loader.get_dataframes(
            ratings_user_id=ratings_user_id,
            # get all rated movies with user_id between 1 and 5, need to filter before training model
            clean_and_index_dfs=clean_and_index_dfs,
            # does not clean or indexes dfs, need to do that before training model
            filtering_type=filtering_type
        )
        # Saves full dfs in session
        st.session_state[rated_df_path] = full_rated_df
        st.session_state[unrated_df_path] = full_unrated_df
    else:
        full_rated_df = st.session_state[rated_df_path]
        full_unrated_df = st.session_state[unrated_df_path]

    return full_rated_df, full_unrated_df


def get_user_ids_from_rated_df(full_rated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get list of unique users from rated dataframe and saved them in session if not already

    :param full_rated_df:
    :return: pd.DataFrame:
    """
    if "user_ids" not in st.session_state:
        # Make temp df
        temp_full_rated_df = full_rated_df.copy()[['userId']]
        # Add column count group by userId
        temp_full_rated_df['numberOfRatedMovies'] = (
            temp_full_rated_df.groupby('userId')['userId'].transform('count')
        )
        temp_full_rated_df = temp_full_rated_df.drop_duplicates()

        if UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF:
            user_ids = (temp_full_rated_df.sort_values(
                by='numberOfRatedMovies',
                ascending=False)
            )
        else:
            user_ids = (temp_full_rated_df.sort_values(
                by='userId',
                ascending=True)
            )
        write_log(
            f"Unique users in app_data_loader: With the most Ratings?: {UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF}, {user_ids}")
        st.session_state["user_ids"] = user_ids
    else:
        user_ids = st.session_state["user_ids"]

    return user_ids


def get_user_ratings_from_rated_df(full_rated_df: pd.DataFrame, selected_user: str) -> Tuple[pd.DataFrame, int]:
    """
    Filter the rated movies for the selected user

    :param full_rated_df:
    :param selected_user:
    :return: Tuple[pd.DataFrame, int]:
    """
    user_ratings_df = full_rated_df[full_rated_df['userId'] == selected_user]
    number_of_unique_rated_movies_by_the_user = user_ratings_df['movieId'].nunique()
    return user_ratings_df, number_of_unique_rated_movies_by_the_user


def clean_and_index_user_ratings_df_and_full_unrated_df(
        user_ratings_df: pd.DataFrame,
        full_unrated_df: pd.DataFrame,
        filtering_type: FilteringType
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans and indexes the dataframes
    :param user_ratings_df:
    :param full_unrated_df:
    :param filtering_type:
    :return:
    """
    rated_df, unrated_df = data_loader._clean_and_index_dfs(user_ratings_df, full_unrated_df, filtering_type)
    return rated_df, unrated_df
