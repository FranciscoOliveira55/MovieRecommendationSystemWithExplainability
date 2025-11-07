from dataclasses import dataclass

import pandas as pd

from configs import FilteringType, ModelType


@dataclass
class UiModelPageDto:
    selected_user: str
    selected_filtering_type: FilteringType
    selected_model_type: ModelType

    #number_of_unique_rated_movies_by_the_user: int
    #movies_rated_by_selected_user_df: pd.DataFrame
    #uncleaned_rated_df: pd.DataFrame

