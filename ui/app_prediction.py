from typing import Union

from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor

from core import prediction
import pandas as pd
import torch.nn as nn
import streamlit as st
from configs import ModelType, FilteringType
import xgboost as xgb

if __name__ == '__main__':
    pass


def make_rating_predictions_for_unrated_movies_df(
        unrated_df: pd.DataFrame,
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        model_type: ModelType,
        filtering_type: FilteringType,
        user_index: pd.Index,  # Only needed for cf embeddings
        item_index: pd.Index
) -> pd.DataFrame:
    """
    Make rating predictions for unrated movies

    :param unrated_df:
    :param model:
    :param model_type:
    :param filtering_type:
    :param user_index:
    :param item_index:
    :return: pd.DataFrame:
    """
    # if "predicted_rating_df" not in st.session_state:
    predicted_rating_df = prediction.predict_ratings_for_unrated_movies(
        unrated_movies_df=unrated_df,
        model=model,
        model_type=model_type,
        filtering_type=filtering_type,
        user_index=user_index,
        item_index=item_index
    )
    #    st.session_state["predicted_rating_df"] = predicted_rating_df
    # else:
    #    predicted_rating_df = st.session_state["predicted_rating_df"]
    return predicted_rating_df
