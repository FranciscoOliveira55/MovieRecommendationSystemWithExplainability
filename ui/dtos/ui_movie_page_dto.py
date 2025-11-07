from dataclasses import dataclass

import pandas as pd

from configs import FilteringType, ModelType


@dataclass
class UiMoviePageDto:
    selected_user: str
    selected_mode: str
    selected_movie_df: pd.DataFrame

    def __post_init__(self):
        # Verify is selected_user is not empty
        if not self.selected_user or not isinstance(self.selected_user, str):
            raise ValueError("selected_user must be a not empty string.")