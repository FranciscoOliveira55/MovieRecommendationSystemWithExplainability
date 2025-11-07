from dataclasses import dataclass

import pandas as pd

from configs import FilteringType, ModelType


@dataclass
class UiUserPageDto:
    selected_user: str
    number_of_unique_rated_movies_by_the_user: int
    movies_rated_by_selected_user_df: pd.DataFrame
    #uncleaned_rated_df: pd.DataFrame

    def __post_init__(self):
        # Verify is selected_user is not empty
        if not self.selected_user:
            raise ValueError("selected_user can't empty.")