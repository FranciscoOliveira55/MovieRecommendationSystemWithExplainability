from dataclasses import dataclass
import random

import pandas as pd


@dataclass
class UiAddMoviePageDto:
    movie_genres: list[str]
    next_unique_available_movie_id:int = random.randint(100000000, 200000000) #Random int between 100M and 200M (low risk of collision)


