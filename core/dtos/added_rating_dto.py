from dataclasses import dataclass

from datetime import datetime


@dataclass
class AddedRatingDTO:
    userId: int
    movieId: int
    rating: float
    timestamp: datetime = datetime.now().replace(microsecond=0)

