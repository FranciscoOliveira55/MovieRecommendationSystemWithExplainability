import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import api_keys
from configs import GetMoviesDetailsFromTmdbApiConfig, FilePaths
from core.utils import write_log, read_df_data_from_csv


def get_movie_details_from_tmdb(tmdb_id: int) -> dict:
    """
    Given a TMDb movie ID and API key, fetch movie details from TheMovieDB.
    Returns a dictionary with title, overview, poster URL, release date, and more.

    :param tmdb_id:
    :return: dict:
    """
    write_log(f"Looking for tmdb details, tmdb_id: {tmdb_id}")

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {
        "api_key": api_keys.THE_MOVIE_DB_API_KEY,
        "language": "en-US"
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        write_log(f"TMDb API request failed, tmdb_id: {tmdb_id} - {response.status_code} - {response.text}")

    movie = response.json()

    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None

    return {
        "title": movie.get("title", None),
        "overview": movie.get("overview", None),
        "release_date": movie.get("release_date", None),
        "poster_url": poster_url,
        "tmdb_id": movie.get("id", None),
        "vote_average": movie.get("vote_average", None),
        "popularity": movie.get("popularity", None)
    }


def get_movies_details_from_tmdb(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a DataFrame of movies (with 'movieId' and 'tmdbId') with TMDb metadata using multithreading.
    Returns a merged DataFrame.

    :param movies_df:
    :return: pd.DataFrame:
    """
    # Drop title column
    movies_df = movies_df.drop(columns=["title"], errors='ignore')

    # Drop lines without tmdbId (none hopefully)
    tmdb_ids = movies_df['tmdbId'].dropna().astype(int).unique()

    if len(tmdb_ids) > 0:
        enriched_data = []

        with ThreadPoolExecutor(
                max_workers=GetMoviesDetailsFromTmdbApiConfig.REQUEST_MULTITHREADING_MAX_WORKERS) as executor:
            future_to_id = {executor.submit(get_movie_details_from_tmdb, tmdb_id): tmdb_id for tmdb_id in tmdb_ids}

            for future in as_completed(future_to_id):
                result = future.result()
                if result:
                    enriched_data.append(result)

        # Convert to DataFrame
        tmdb_details_df = pd.DataFrame(enriched_data)

        # Merge on tmdbId
        enriched_movies_df = pd.merge(movies_df, tmdb_details_df, left_on='tmdbId', right_on='tmdb_id', how='left')

        # Drop rows with none values
        enriched_movies_df = enriched_movies_df.dropna(
            subset=['tmdb_id', 'title', 'overview', 'release_date', 'poster_url', 'vote_average', 'popularity'],
            how='any'
        )
    else:
        enriched_movies_df = pd.DataFrame()

    # Added movies
    added_movies_tmdb_details: pd.DataFrame = read_df_data_from_csv(FilePaths.ADDED_MOVIES_TMDB_DETAILS_CSV_PATH,
                                                                    ("movieId", "title", "release_date", "vote_average",
                                                                     "popularity", "overview", "poster_url"
                                                                     ))
    added_movies_with_tmdb_details = pd.merge(movies_df, added_movies_tmdb_details, on="movieId", how="inner")

    #Concat both dfs
    enriched_movies_df = pd.concat(
        [enriched_movies_df,
         added_movies_with_tmdb_details
         ], ignore_index=True
    )
    return enriched_movies_df
