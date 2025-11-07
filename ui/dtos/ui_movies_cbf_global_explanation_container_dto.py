from dataclasses import dataclass
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor
import xgboost as xgb


from configs import FilteringType, ModelType


@dataclass
class UiMoviesCbfGlobalExplanationContainerDto:
    selected_model_type: ModelType
    prediction_model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]
    rated_df:pd.DataFrame
    predicted_movies_to_explain_df: pd.DataFrame

