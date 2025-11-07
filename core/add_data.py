import os

import pandas as pd
from configs import FilePaths
from core.dtos.added_movie_dto import AddedMovieDTO
from core.dtos.added_rating_dto import AddedRatingDTO
from core.dtos.added_user_dto import AddedUserDTO

if __name__ == '__main__':
    pass
    print("Hi there, you're in the add_data module :)")
    print("This module uses dtos of new users, interactions or items and adds them to the correspondent file :)")


def add_movie_to_csv(added_movie_dto: AddedMovieDTO):
    """
    Adds the information in added_movie_dto to csv file.

    :param added_movie_dto:
    :return:
    """
    # Turns DTO into a dataframe/dictionary
    movie_data = {
        "movieId": added_movie_dto.movieId,
        "title": added_movie_dto.title,
        "genres": "|".join(added_movie_dto.genres)  # usa formato tipo MovieLens
    }
    # Adds the movie to the csv file
    df_new = pd.DataFrame([movie_data])
    file_path = FilePaths.ADDED_MOVIES_CSV_PATH
    df_new.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

    movie_tmdb_details_data = {
        "movieId": added_movie_dto.movieId,
        "title": added_movie_dto.title,
        "release_date": added_movie_dto.release_date,
        "vote_average": added_movie_dto.vote_average,
        "popularity": added_movie_dto.popularity,
        "overview": added_movie_dto.overview,
        "poster_url": added_movie_dto.poster_url
    }
    # Adds the movie tmdb details to the csv file
    df_new_1 = pd.DataFrame([movie_tmdb_details_data])
    file_path_1 = FilePaths.ADDED_MOVIES_TMDB_DETAILS_CSV_PATH
    df_new_1.to_csv(file_path_1, mode='a', header=not os.path.exists(file_path_1), index=False)


def add_user_to_csv(added_user_dto: AddedUserDTO):
    """
    Adds the information in added_user_dto to csv file.

    :param added_user_dto:
    :return:
    """
    # Turns DTO into a dataframe/dictionary
    user_data = {
        "userId": added_user_dto.userId,
    }
    # Adds the user to the csv file
    df_new = pd.DataFrame([user_data])
    file_path = FilePaths.ADDED_USERS_CSV_PATH
    df_new.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)


def add_rating_to_csv(added_rating_dto: AddedRatingDTO):
    """
    Adds the information in added_rating_dto to csv file.

    :param added_rating_dto:
    :return:
    """
    # Turns DTO into a dataframe/dictionary
    rating_data = {
        "userId": added_rating_dto.userId,
        "movieId": added_rating_dto.movieId,
        "rating": added_rating_dto.rating,
        "timestamp": added_rating_dto.timestamp,
    }
    # Adds the rating to the csv file
    df_new = pd.DataFrame([rating_data])
    file_path = FilePaths.ADDED_RATINGS_CSV_PATH
    df_new.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
