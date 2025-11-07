from dataclasses import dataclass


@dataclass
class AddedMovieDTO:
    movieId: int
    title: str
    genres: list[str]

    release_date:any
    vote_average:float
    popularity:float
    overview:str
    poster_url:str