import numpy as np
from streamlit import columns

import configs
from typing import Tuple

import pandas as pd
from configs import FilePaths, UserPickConfig, FilteringType
from core.utils import write_log, read_df_data_from_csv, save_df_locally_to_csv

if __name__ == '__main__':
    pass
    print("Hi there, you're in the data_loader module :)")
    print("This module loads and prepares movie's data in dataframes :)")


def add_links_to_movies_df(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the tmdbId to the movies in movies_df

    :param movies_df:
    :return: pd.DataFrame:
    """

    # Read the links.csv file
    links_df = read_df_data_from_csv(FilePaths.LINKS_CSV_PATH, columns_names=("movieId", "tmdbId"))  # , "imdbId"

    # Perform the left merge on movieId
    merged_df = pd.merge(movies_df, links_df, on="movieId", how="left")

    return merged_df


def get_dataframes(
        ratings_user_id: int = 1,
        clean_and_index_dfs: bool = True,
        filtering_type: FilteringType = FilteringType.CONTENT_BASED_FILTERING,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the dataframes with the rated and unrated movies

    :param ratings_user_id:
    :param clean_and_index_dfs:
    :param filtering_type:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """

    match filtering_type.value:
        case FilteringType.CONTENT_BASED_FILTERING.value:
            # rated_movies_df: rated_df filtered only to have movies of 1 or X user_ids  [user_id, movie_id, movie_features, rating]
            # unrated_movies_df: all movies not rated by 1 or X user_ids                 [movie_id, movie_features](not rated by 1 or X user_ids)
            rated_movies_df, unrated_movies_df = _get_dataframes_for_cbf(ratings_user_id, clean_and_index_dfs)
            return rated_movies_df, unrated_movies_df
        case FilteringType.COLLABORATIVE_FILTERING.value:
            # sample_ratings_df: rated_df filtered only to have X user_ids [user_id, movie_id, rating]
            # unrated_movies_df: all movies not rated by the selected user_id       [selected_user_id, movie_id] (not rated by 1 user_id, but rated by the others)
            sample_ratings_df, unrated_movies_df = _get_dataframes_for_cf(ratings_user_id, clean_and_index_dfs)
            return sample_ratings_df, unrated_movies_df
        case FilteringType.HYBRID_FILTERING.value:
            # sample_hybrid_rated_df: rated_df filtered only to have X user_ids [user_id, movie_id, rating, features*]
            # unrated_movies_df: all movies not rated by the selected user_id (doesn't matter if it's rated or not by other users)
            sample_hybrid_rated_df, hybrid_unrated_df = _get_dataframes_for_hybrid(ratings_user_id, clean_and_index_dfs)
            return sample_hybrid_rated_df, hybrid_unrated_df
        case _:
            raise ValueError(f"Unsupported filtering type: {filtering_type}, type={type(filtering_type)}")


def _get_dataframes_for_cbf(
        selected_user_id: int = 1,  # rated by this user(s), unrated by this user(s)
        clean_and_index_dfs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the dataframes with the rated and unrated movies for cbf

    :param selected_user_id:
    :param clean_and_index_dfs:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Make file paths
    transformed_rated_movies_df_path, transformed_unrated_movies_df_path = configs.FilePaths.get_transformed_df_path(
        filtering_type=FilteringType.CONTENT_BASED_FILTERING,
        selected_user_id=selected_user_id,
        clean_and_index_dfs=clean_and_index_dfs
    )

    # Try to read transformed arrays
    try:
        write_log("Try to read transformed csv files...")
        rated_movies_df = read_df_data_from_csv(transformed_rated_movies_df_path)
        unrated_movies_df = read_df_data_from_csv(transformed_unrated_movies_df_path)
        write_log("Transformed csv files successfully read")
    except FileNotFoundError:
        write_log("Transformed csv files not found, reading original ones...")
        # Read movieId, title and genres from movie.csv
        movies_df = read_df_data_from_csv(FilePaths.MOVIES_CSV_PATH, ("movieId", "title", "genres"))
        write_log(f"movies.csv has: {movies_df['movieId'].nunique()} movies")
        # Read userId, movieId, timestamp and rating from rating.csv
        ratings_df = read_df_data_from_csv(FilePaths.RATINGS_CSV_PATH, ("userId", "movieId", "rating", "timestamp"))
        write_log(f"ratings.csv has: {ratings_df['userId'].nunique()} users")
        # Filter rows only for the ratings from userId == X (or 5 with most ratings) if userId < 1 (light weight)
        user_ratings_df = _filter_ratings_df(ratings_df, selected_user_id, FilteringType.CONTENT_BASED_FILTERING)
        # Left Join movies_df with user_ratings_df in a single dataframe with movies_df.movieId = user_ratings_df.movieId
        # Remove movies with "(no genres listed)"
        merged_df = _merge_movies_with_ratings(movies_df, user_ratings_df)
        # Turns 1 column genres with a str of the genres into a series of columns, 1 for each genre, each with binary values (1 or 0)
        # (df[genres] = "horror|comedy" -> df[horror] = 1, df[comedy] = 1, ...)
        merged_with_processed_genres = _preprocess_genres(merged_df)
        # Split dataframe between rated and unrated movies (where rating is or is not none)
        rated_movies_df, unrated_movies_df = _split_rated_unrated(merged_with_processed_genres)
        # Drop useless columns and indexed dataframes
        if clean_and_index_dfs:
            rated_movies_df, unrated_movies_df = _clean_and_index_dfs(rated_movies_df, unrated_movies_df,
                                                                      FilteringType.CONTENT_BASED_FILTERING)
        # Save transformed dfs
        save_df_locally_to_csv(rated_movies_df, transformed_rated_movies_df_path)
        save_df_locally_to_csv(unrated_movies_df, transformed_unrated_movies_df_path)
        write_log("Datasets successfully read, transformed and saved locally")

    # get added movies and interactions of the selected user df
    added_rated_movies_df, added_unrated_movies_df = _get_added_dataframes_for_cbf(selected_user_id, clean_and_index_dfs)

    # add added interactions to the rated,unrated df
    rated_movies_df = pd.concat(
        [rated_movies_df,
         added_rated_movies_df
         ], ignore_index=True
    )
    # add added movies to the unrated df
    unrated_movies_df = pd.concat(
        [unrated_movies_df,
         added_unrated_movies_df
         ], ignore_index=True
    )
    if clean_and_index_dfs:
        unrated_movies_df = unrated_movies_df[~unrated_movies_df.index.isin(rated_movies_df.index)]
    else:
        unrated_movies_df = unrated_movies_df[~unrated_movies_df["movieId"].isin(rated_movies_df["movieId"])]

    # Return dataframes
    return rated_movies_df, unrated_movies_df


def _get_added_dataframes_for_cbf(
        selected_user_id: int = 1,  # rated by this user(s), unrated by this user(s)
        clean_and_index_dfs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the dataframes with the rated and unrated added movies for cbf

    :param selected_user_id:
    :param clean_and_index_dfs:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """

    write_log("Reading added dfs")
    # Read movieId, title and genres from movie.csv
    added_movies_df = read_df_data_from_csv(FilePaths.ADDED_MOVIES_CSV_PATH, ("movieId", "title", "genres"))
    write_log(f"added_movies.csv has: {added_movies_df['movieId'].nunique()} movies")
    # Read userId, movieId, timestamp and rating from rating.csv
    added_ratings_df = read_df_data_from_csv(FilePaths.ADDED_RATINGS_CSV_PATH, ("userId", "movieId", "rating", "timestamp"))
    write_log(f"added_ratings.csv has: {added_ratings_df['userId'].nunique()} users")
    # Filter rows only for the ratings from userId == X (or 5 with most ratings) if userId < 1 (light weight)
    added_user_ratings_df = _filter_ratings_df(added_ratings_df, selected_user_id, FilteringType.CONTENT_BASED_FILTERING)

    #Grab movieIds of added_user_ratings_df
    movie_ids = added_user_ratings_df["movieId"].unique()

    #Grab movies with those movieIds (if any)
    movies_df = read_df_data_from_csv(FilePaths.MOVIES_CSV_PATH, ("movieId", "title", "genres"))
    movies_df = movies_df[movies_df["movieId"].isin(movie_ids)]

    #Concatenate with added added_movies_df
    added_movies_df = pd.concat(
        [added_movies_df,
         movies_df
         ], ignore_index=True
    )

    # Outer Join added_movies_df with added_user_ratings_df in a single dataframe with movies_df.movieId = user_ratings_df.movieId
    merged_df = pd.merge(added_movies_df, added_user_ratings_df, on="movieId", how="left")

    # Remove the movies with "(no genres listed)"
    merged_df = merged_df[merged_df["genres"] != "(no genres listed)"]

    # Turns 1 column genres with a str of the genres into a series of columns, 1 for each genre, each with binary values (1 or 0)
    # (df[genres] = "horror|comedy" -> df[horror] = 1, df[comedy] = 1, ...)
    merged_with_processed_genres = _preprocess_genres(merged_df)
    # Split dataframe between rated and unrated movies (where rating is or is not none)
    rated_movies_df, unrated_movies_df = _split_rated_unrated(merged_with_processed_genres)
    # Drop useless columns and indexed dataframes

    if clean_and_index_dfs:
        rated_movies_df, unrated_movies_df = _clean_and_index_dfs(rated_movies_df, unrated_movies_df,
                                                                  FilteringType.CONTENT_BASED_FILTERING)
    # Return dataframes
    return rated_movies_df, unrated_movies_df


def _get_dataframes_for_cf(
        selected_user_id: int = 1,  # rated by this user, unrated by this user (but rated by others)
        clean_and_index_dfs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the dataframes with the rated and unrated movies for cf

    :param selected_user_id:
    :param clean_and_index_dfs:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """

    # Make file paths
    transformed_rated_movies_df_path, transformed_unrated_movies_df_path = configs.FilePaths.get_transformed_df_path(
        filtering_type=FilteringType.COLLABORATIVE_FILTERING,
        selected_user_id=selected_user_id,
        clean_and_index_dfs=clean_and_index_dfs
    )

    # Try to read transformed arrays
    try:
        write_log("Try to read transformed csv files...")
        sample_ratings_df = read_df_data_from_csv(transformed_rated_movies_df_path)
        unrated_movies_df = read_df_data_from_csv(transformed_unrated_movies_df_path)
        write_log("Transformed csv files successfully read")
    except FileNotFoundError:
        write_log("Transformed csv files not found, reading original ones...")
        # Read userId, movieId, timestamp and rating from rating.csv
        ratings_df: pd.DataFrame = read_df_data_from_csv(FilePaths.RATINGS_CSV_PATH,
                                                         ("userId", "movieId", "rating", "timestamp"))
        # Read movieId, title and genres from movie.csv
        movies_df: pd.DataFrame = read_df_data_from_csv(FilePaths.MOVIES_CSV_PATH, ("movieId", "title"))
        write_log(f"ratings.csv has: {ratings_df['userId'].nunique()} users")
        # Filter rows only for X number of users (with selected user)
        sample_ratings_df = _filter_ratings_df(ratings_df, selected_user_id, FilteringType.COLLABORATIVE_FILTERING)
        # Add movie title
        sample_ratings_df = pd.merge(sample_ratings_df, movies_df, on="movieId", how="left")
        # Grab the movies rated by the selected user
        user_rated_movie_ids = sample_ratings_df[sample_ratings_df['userId'] == selected_user_id]['movieId'].unique()
        # Unrated df is a subset of ratings df with the movies rated by other users than the selected user
        other_users_ratings_df = sample_ratings_df[sample_ratings_df['userId'] != selected_user_id]
        # Remove the movies rated by both other and the selected user
        filtered_other_users_ratings_df = other_users_ratings_df[
            ~other_users_ratings_df['movieId'].isin(user_rated_movie_ids)]
        # Grab only the unique movie IDs and title
        other_users_ratings_unique_movie_ids = filtered_other_users_ratings_df[['movieId', 'title']].drop_duplicates(
            subset='movieId').reset_index(drop=True)

        # Turn unrated into 2 columns [ selected_user_id, movie_id]
        unrated_movies_df = pd.DataFrame({
            'userId': [selected_user_id] * len(other_users_ratings_unique_movie_ids),
            'movieId': other_users_ratings_unique_movie_ids['movieId'],
            'title': other_users_ratings_unique_movie_ids['title']
        })

        # Drop useless columns from rated_df
        if clean_and_index_dfs:
            sample_ratings_df, _ = _clean_and_index_dfs(sample_ratings_df, sample_ratings_df,
                                                        FilteringType.COLLABORATIVE_FILTERING)

        # Save transformed dfs
        save_df_locally_to_csv(sample_ratings_df, transformed_rated_movies_df_path)
        save_df_locally_to_csv(unrated_movies_df, transformed_unrated_movies_df_path)
        write_log("Datasets successfully read, transformed and saved locally")
    # Return dataframes
    return sample_ratings_df, unrated_movies_df


def _get_dataframes_for_hybrid(
        selected_user_id: int = 1,  # rated by this user, unrated by this user
        clean_and_index_dfs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the dataframes with the rated and unrated movies for hybrid filtering

    :param selected_user_id:
    :param clean_and_index_dfs:
    :return: Tuple[pd.DataFrame, pd.DataFrame]:
    """

    # hybrid_rated_df: [userId, userEmbedding[*], movieId, movieEmbedding[*], ratings]
    # hybrid_unrated_df: [selected_user, selected_userEmbedding[*], movieId, movieEmbedding[*]]

    # Get ALL movies with preprocessed genres
    cbf_rated_df, cbf_unrated_df = get_dataframes(
        ratings_user_id=selected_user_id,
        clean_and_index_dfs=False,
        filtering_type=FilteringType.CONTENT_BASED_FILTERING
    )
    # Get item embeddings (ALL movie's features)
    cbf_df_union = pd.concat([cbf_rated_df, cbf_unrated_df], ignore_index=True)
    # write_log(f"cbf_df_union.columns: {cbf_df_union.columns}")
    # Get rated_df for cf ([userId, itemId, rating] of 100 users) (all interactions of 100 users)
    cf_rated_df, _ = get_dataframes(
        ratings_user_id= 1, #selected_user_id,
        clean_and_index_dfs=False,
        filtering_type=FilteringType.COLLABORATIVE_FILTERING
    )
    # write_log(f"cf_rated_df.columns: {cf_rated_df.columns}")

    # Add cbf interactions (include the "Added ones" of the selected user) to the cf ones
    cf_rated_df = pd.concat(
        [cf_rated_df,
         cbf_rated_df[["userId", "movieId", "rating", "title", "timestamp"]]
         ], ignore_index=True
    )

    # Drop common columns except movieId (preparing for Join)
    cbf_df_union = cbf_df_union.drop(columns=["userId", "rating", "timestamp"],
                                     errors="ignore")  # (keep only movieId,features*)
    cf_rated_df = cf_rated_df.drop(columns=["title"], errors="ignore")  # (keep only interactions)
    # Drop duplicated lines
    cbf_df_union = cbf_df_union.drop_duplicates()
    cf_rated_df = cf_rated_df.drop_duplicates()

    # Join cbf_df_union with cf_rated_df using left join (preserve all movies)
    # (features of ALL movies + interactions of 100 users (including the selected one))
    df_hybrid = pd.merge(cbf_df_union, cf_rated_df, on=['movieId'], how="left")
    # write_log(f"df_hybrid.columns: {df_hybrid.columns}")

    # Split in rated and unrated
    # MovieIds rated by the selected user
    rated_movie_ids_by_the_selected_user = df_hybrid[df_hybrid["userId"] == selected_user_id]["movieId"].unique()
    # Rated_df has the Movies rated by ANY user (any of the selected user or the X chosen by the cf)
    hybrid_rated_df = df_hybrid[df_hybrid["rating"].notna()].copy()
    # Unrated_df has the Movies NOT rated BY the selected user
    # (YES, movies rated by other users that not the selected one, are present in both rated and unrated dfs)
    hybrid_unrated_df = df_hybrid[~df_hybrid["movieId"].isin(rated_movie_ids_by_the_selected_user)].copy()
    # write_log(f"There are {cbf_unrated_df["movieId"].nunique()} movies, not rated by the selected user")
    # write_log(f"There are {hybrid_unrated_df["movieId"].nunique()} movies, not rated by the selected user")

    # Calc user embeddings
    item_features = df_hybrid.columns
    item_features = item_features.drop(labels=["userId", "movieId", "rating", "title", "timestamp", "imdbId", "tmdbId"],
                                       errors="ignore")
    # write_log(f"item_features: {item_features}")
    max_rating = 5
    temp_hybrid_rated_df = hybrid_rated_df.copy()
    temp_hybrid_rated_df['rating_norm'] = temp_hybrid_rated_df['rating'] / max_rating
    user_features = []

    for col in item_features:
        user_feature = f"user{col}"
        temp_hybrid_rated_df[user_feature] = temp_hybrid_rated_df[col] * temp_hybrid_rated_df['rating_norm']
        user_features.append(user_feature)

    user_embeddings = temp_hybrid_rated_df.groupby('userId')[user_features].mean()  # This makes userId index
    # Normalization
    user_embeddings = user_embeddings.div(user_embeddings.sum(axis=1), axis=0)
    # Reset index
    user_embeddings = user_embeddings.reset_index()
    # Join item embeddings to user embeddings
    hybrid_rated_df = pd.merge(hybrid_rated_df, user_embeddings, on=['userId'], how="left")

    # Add selected user embedding to unrated df
    selected_user_embedding = user_embeddings[user_embeddings['userId'] == selected_user_id]

    for col in selected_user_embedding.columns:
        hybrid_unrated_df[col] = selected_user_embedding[col].values[0]
    hybrid_unrated_df = hybrid_unrated_df.drop(columns=["rating", "timestamp"], errors="ignore")
    hybrid_unrated_df = hybrid_unrated_df.drop_duplicates()

    # Clear dfs if necessary
    if clean_and_index_dfs:
        hybrid_rated_df, hybrid_unrated_df = _clean_and_index_dfs(hybrid_rated_df, hybrid_unrated_df,
                                                                  FilteringType.HYBRID_FILTERING)

    # Return dfs
    return hybrid_rated_df, hybrid_unrated_df


def get_added_users_dataframe(
) -> pd.DataFrame:
    """
    Returns the dataframe with the added user ids

    :return: pd.DataFrame:
    """
    write_log("Reading added users df")
    # Read userId from added_users.csv
    added_users_df = pd.read_csv(FilePaths.ADDED_USERS_CSV_PATH, usecols=["userId"])
    return added_users_df


def _filter_ratings_df(
        ratings_df: pd.DataFrame,
        selected_user_id: int = 1,
        filtering_type: FilteringType = FilteringType.CONTENT_BASED_FILTERING
) -> pd.DataFrame:
    """
    Filters ratings dataframe depending on filtering type

    :param ratings_df:
    :param selected_user_id:
    :param filtering_type:
    :return: pd.DataFrame:
    """
    # If user_id is a positive number, filter df only for that user
    if (selected_user_id >= 1) and (filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value):
        return ratings_df[ratings_df["userId"] == selected_user_id]
    # If user_id is a negative number, filter either by 1:X users or top X user with most ratings
    else:  # if selected_user_id < 1 # (-1)
        # Make df with the count of ratings by user_id
        top_users_counts = ratings_df['userId'].value_counts()  # Get series (userId, counts)
        top_users_df = top_users_counts.reset_index()  # Turns series into DataFrame
        top_users_df.columns = ['userId', 'movie_count']
        # Get pick_users_with_most_ratings and number_of_x_top_users values
        if filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value:
            pick_users_with_most_ratings = UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF
            number_of_x_top_users = UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF
        else:
            pick_users_with_most_ratings = UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF
            number_of_x_top_users = UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF
        # Grab either first X users or X users with most ratings
        if pick_users_with_most_ratings:
            # Get top X users with the most ratings
            top_users_df = top_users_df[0:number_of_x_top_users]  # X first rows
        else:
            # Get users with ID 1-X
            top_users_df = ratings_df[
                ratings_df['userId'].between(1, number_of_x_top_users)]
            top_users_df = (
                top_users_df.groupby('userId')
                .size()
                .reset_index(name='movie_count')
            )
        log_message = f"Top {number_of_x_top_users} users:" + "With the most Ratings?:" + str(
            pick_users_with_most_ratings) + f"filtering_type: {filtering_type.value}" + " ".join(
            f" || User {user_id} -> {count} ratings" for user_id, count in
            zip(top_users_df['userId'], top_users_df['movie_count']))
        write_log(log_message)
        # Return ratings of x top users including selected user
        filtered_ratings_df = ratings_df[
            ratings_df['userId'].isin(top_users_df['userId'].tolist() + [selected_user_id])]
        write_log(str(filtered_ratings_df.columns))
        return filtered_ratings_df


def _merge_movies_with_ratings(movies_df: pd.DataFrame, user_ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges movies dataframe with user ratings using movieID (left join (keep all the movies, even the unrated ones))

    :param movies_df:
    :param user_ratings_df:
    :return: pd.DataFrame:
    """
    # Joins moved with user ratings, keeping the unrated movies as well
    merged_df = pd.merge(movies_df, user_ratings_df, on="movieId", how="left")
    # Remove the movies without genres
    merged_df = merged_df[merged_df["genres"] != "(no genres listed)"]
    # Return the dataframe
    return merged_df


def _preprocess_genres(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes genre information: explode + one-hot encoding + merge back

    :param merged_df:
    :return: pd.DataFrame:
    """
    # Turns a str with genres into a list of strs of genres ("Horror|Comedy" -> ["Horror", "Comedy"])
    merged_df["genre_list"] = merged_df["genres"].str.split("|")
    # Explodes each genre to a different row    ("Adventure"\n, "Animation"\n,"Children"\n) (all in the genre_list column)
    exploded = merged_df.explode("genre_list")
    # Turns each genre row into a column with 1 or 0 ( 1   0   0   0)
    # df[genre_list] = "horror" -> df[horror] = 1
    dummies = pd.get_dummies(exploded["genre_list"])
    # Adds movieId to the dummies, so we can make a join

    # Forces genre column if it doesnt exist
    forced_genres:list[str] = ["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
    for genre in forced_genres:
        if genre not in dummies.columns:
            dummies[genre] = 0  # adds column with value 0

    dummies["movieId"] = exploded["movieId"]
    # Group rows by movieId ( 1   0   1   0) (we max the values of different genre rows (of the same movie))
    genre_df = dummies.groupby("movieId").max()
    genre_df = genre_df.astype(int) #Use int8 intead?

    # Add genre columns to the original dataframe
    merged_df = pd.merge(merged_df, genre_df, on="movieId", how="inner")
    # Drop the old genre columns
    merged_df = merged_df.drop(columns=["genres", "genre_list"])
    # Return the processed dataframe
    return merged_df


def _split_rated_unrated(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Splits the dataframe into rated and unrated

    :param merged_df:
    :return: tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Rated movie is a row that "rating" is a value (not none)
    rated_df = merged_df[merged_df["rating"].notna()]
    # Unrated movie is a row that "rating" is not a value (is none)
    unrated_df = merged_df[merged_df["rating"].isna()]
    # Return dataframes
    return rated_df, unrated_df


def _clean_and_index_dfs(
        rated_df: pd.DataFrame,
        unrated_df: pd.DataFrame,
        filtering_type: FilteringType
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans and indexes dataframes

    :param rated_df:
    :param unrated_df:
    :param filtering_type:
    :return: tuple[pd.DataFrame, pd.DataFrame]:
    """
    # Drops unnecessary columns for training and sets index on movieId
    # unnecessary_columns = ["title", "userId", "timestamp", "rating"]
    # rated_df = rated_df.drop(columns=unnecessary_columns, errors='ignore')
    # unrated_df = unrated_df.drop(columns=unnecessary_columns, errors='ignore')
    match filtering_type.value:
        case FilteringType.CONTENT_BASED_FILTERING.value | FilteringType.HYBRID_FILTERING.value:
            rated_df = rated_df.drop(
                columns=["title", "userId", "timestamp", "imdbId", "tmdbId", "index"], errors='ignore')
            unrated_df = unrated_df.drop(
                columns=["title", "userId", "timestamp", "rating", "imdbId", "tmdbId", "index"], errors='ignore')
            rated_df = rated_df.set_index("movieId")
            unrated_df = unrated_df.set_index("movieId")
        case FilteringType.COLLABORATIVE_FILTERING.value:
            rated_df = rated_df.drop(columns=["title", "timestamp", "imdbId", "tmdbId", "index"], errors='ignore')
            unrated_df = unrated_df.drop(
                columns=["title", "timestamp", "rating", "imdbId", "tmdbId", "index"], errors='ignore')
        case _:
            raise ValueError(f"Unsupported filtering type: {filtering_type}, type={type(filtering_type)}")

    # Return dataframes
    return rated_df, unrated_df
