from dataclasses import dataclass
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor
import xgboost as xgb


from configs import FilteringType, ModelType


@dataclass
class UiMovieCfLocalExplanationContainerDto:

    selected_user: str
    prediction_model: Union[NeuralNetRegressor]  # Trained model.
    rated_df_used_in_training: pd.DataFrame
    user_index: pd.Index  # Indexes of users collected in training
    predicted_movie_to_explain_df: pd.DataFrame