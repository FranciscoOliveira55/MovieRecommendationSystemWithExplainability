from typing import Union, Tuple

import streamlit as st
from skorch import NeuralNetRegressor

from core import model
import pandas as pd
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from configs import ModelType, FilteringType

if __name__ == '__main__':
    pass


def generate_model(rated_df: pd.DataFrame, model_type: ModelType, filtering_type: FilteringType, user_id: int) \
        -> Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        Union[pd.Index, None], Union[pd.Index, None]]:
    """
    Generate the prediction models

    :param rated_df:
    :param model_type:
    :param filtering_type:
    :param user_id:
    :return: Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        Union[pd.Index, None], Union[pd.Index, None]]:
    """
    if "prediction_model" not in st.session_state:
        prediction_model, user_index, item_index = model.read_or_create_model(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=model_type,
            filtering_type=filtering_type,
            user_id=user_id,
        )
        st.session_state["prediction_model"] = prediction_model
        if filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
            st.session_state["user_index"] = user_index
            st.session_state["item_index"] = item_index
        else:
            user_index = None
            item_index = None
    else:
        prediction_model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor] = st.session_state[
            "prediction_model"]
        if filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
            user_index = st.session_state["user_index"]
            item_index = st.session_state["item_index"]
        else:
            user_index = None
            item_index = None
    return prediction_model, user_index, item_index
