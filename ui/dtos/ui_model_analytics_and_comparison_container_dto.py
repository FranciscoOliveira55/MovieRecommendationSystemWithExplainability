from dataclasses import dataclass
from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor
import xgboost as xgb
from configs import FilteringType, ModelType


@dataclass
class UiModelAnalyticsAndComparisonContainerDto:
    selected_user: str
    selected_filtering_type: FilteringType
    selected_model_type: ModelType

    rated_df:pd.DataFrame
    prediction_model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]



    #number_of_unique_rated_movies_by_the_user: int
    #movies_rated_by_selected_user_df: pd.DataFrame
    #uncleaned_rated_df: pd.DataFrame

