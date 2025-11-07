from dataclasses import dataclass
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor
import xgboost as xgb


from configs import FilteringType, ModelType


@dataclass
class UiMovieHybLocalExplanationContainerDto:

    selected_user: str
    rated_uncleaned_df: pd.DataFrame
    predicted_movie_to_explain: pd.DataFrame