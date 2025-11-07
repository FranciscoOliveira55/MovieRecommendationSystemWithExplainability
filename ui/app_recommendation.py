from core import recommendation
import pandas as pd

if __name__ == '__main__':
    pass


def recommend_the_top_unrated_movies(predicted_rating_df: pd.DataFrame, x_number_of_movies_to_recommend:int) -> pd.DataFrame:
    """
    # Recommend the top unrated movies with the highest predicted rating

    :param predicted_rating_df:
    :param x_number_of_movies_to_recommend:
    :return: pd.DataFrame:
    """
    recommended_movies = recommendation.recommend_x_unrated_movies_with_the_highest_predicted_ratings(
        predicted_df=predicted_rating_df,
        x_number_of_movies_to_recommend=x_number_of_movies_to_recommend
    )
    return recommended_movies


def get_recommended_movies_df_with_full_details(recommended_movies_df: pd.DataFrame,
                                                full_unrated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the titles of the recommended unrated movies, set indexes on movieId and return with the same column order

    :param recommended_movies_df:
    :param full_unrated_df:
    :return:
    """
    recommended_movies_with_full_details = pd.merge(
        recommended_movies_df,
        full_unrated_df[["movieId", "title"]],
        on="movieId",
        how="inner"
    )
    #Make sure it has index on movieId
    indexed_recommended_movies_df = recommended_movies_df.copy()
    indexed_recommended_movies_df = indexed_recommended_movies_df.reset_index()
    indexed_recommended_movies_df = indexed_recommended_movies_df.set_index("movieId")

    # Set movieId as index
    recommended_movies_with_full_details = recommended_movies_with_full_details.set_index("movieId")
    # Makes sure, movies with full details keeps the order
    recommended_movies_with_full_details = recommended_movies_with_full_details.loc[indexed_recommended_movies_df.index]
    return recommended_movies_with_full_details
