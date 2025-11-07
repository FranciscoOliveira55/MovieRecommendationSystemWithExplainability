import pandas as pd

if __name__ == '__main__':
    pass
    print("Hi there, you're in the recommendation module :)")
    print("This module selects and returns the top movie recommendations :)")


def recommend_x_unrated_movies_with_the_highest_predicted_ratings(
        predicted_df: pd.DataFrame,
        x_number_of_movies_to_recommend: int = 1
) -> pd.DataFrame:
    """
    Recommend the top X unrated movies with the highest predicted ratings

    :param predicted_df:
    :param x_number_of_movies_to_recommend:
    :return: pd.DataFrame:
    """
    # Sorts for descending order of value using PredictedRating column and returns the top x rows (x movies with the highest predicted rating)
    top_x_movies = predicted_df.nlargest(min(x_number_of_movies_to_recommend,predicted_df.size), 'PredictedRating')
    return top_x_movies
