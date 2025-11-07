from dataclasses import dataclass

import pandas as pd

from configs import FilteringType, ModelType


@dataclass
class UiMoviesPageDto:
    selected_user: str
    selected_filtering_type: FilteringType
    selected_model_type: ModelType
    selected_mode: str
    predicted_rating_df: pd.DataFrame

    def __post_init__(self):
        # Verify is selected_user is not empty
        if not self.selected_user or not isinstance(self.selected_user, str):
            raise ValueError("selected_user must be a not empty string.")