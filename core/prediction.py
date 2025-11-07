from typing import Union
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from skorch import NeuralNetRegressor
from streamlit import exception

from configs import ModelType, FilteringType
from core.utils import write_log

if __name__ == '__main__':
    pass
    print("Hi there, you're in the prediction module :)")
    print("This module generates rating predictions for unrated movies :)")


def predict_rating_for_movie(
        unrated_movie_series: pd.Series,
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        model_type: ModelType) -> float:
    """
    Predicts the rating of 1 unrated movie using a trained model and an input series (is a wrapper function)

    :param unrated_movie_series:
    :param model:
    :param model_type:
    :return: float:
    """
    # Turns unrated movie series into a ndarray
    unrated_movie_array = _prepare_prediction_data(unrated_movie_series)
    # Predicts unrated movie rating using the model
    predicted_unrated_movie_rating = _predict_rating_for_unrated_movie_with_model_from_ndarray_to_ndarray(
        unrated_movie_array, model, model_type)
    # Returns the predicted rating
    return float(predicted_unrated_movie_rating[0])


def predict_ratings_for_unrated_movies(
        unrated_movies_df: pd.DataFrame,
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        model_type: ModelType,
        filtering_type: FilteringType = FilteringType.CONTENT_BASED_FILTERING,
        user_index: pd.Index = None,  # Only needed for cf embeddings
        item_index: pd.Index = None  # Only needed for cf embeddings
) -> pd.DataFrame:
    """
    Predicts ratings for ALL the movies in a DataFrame (of unrated movies).

    :param unrated_movies_df:
    :param model:
    :param model_type:
    :param filtering_type:
    :param user_index:
    :param item_index:
    :return: pd.DataFrame:
    """
    # Creates a copy of the input DataFrame so the original is not modified
    predicted_df = unrated_movies_df.copy()

    write_log(
        f"Predicting ratings for unrated_movies, filtering_type:{filtering_type}, model_type:{model_type}, unrated_movies_df_size:{predicted_df.shape[0]}...")

    if (filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value) or (filtering_type.value == FilteringType.HYBRID_FILTERING.value):
        # Use vector to predict in batch instead of 1 line at a time (a lot faster)
        input_array: np.ndarray = predicted_df.to_numpy()
    else:
        if (user_index is None) or (item_index is None):
            raise Exception(
                "U need to pass user and item training indexes to predict with a CF model that uses embeddings")
        write_log("Converting ids to embedding indices")
        input_array: np.ndarray = _convert_to_embedding_indices(
            unrated_df=predicted_df.reset_index(),
            user_index=user_index,
            item_index=item_index,
            user_col="userId",
            item_col="movieId"
        )
        write_log("Converted ids to embedding indices successfully")

    write_log("Making predictions")
    # Predict ratings for unrated movies
    predictions = _predict_rating_for_unrated_movie_with_model_from_ndarray_to_ndarray(
        input_array,
        model,
        model_type
    )
    predicted_df["PredictedRating"] = predictions
    # Apply the prediction function to each row (to each movie) of the original dataframe
    # predicted_df["PredictedRating"] = unrated_movies_df.apply(
    #    lambda row: predict_rating_for_movie(row, model, model_type), axis=1
    # )
    write_log(f"Predicting ratings for unrated_movies, completed successfully")
    # Returns dataframe with all the rating predictions done
    return predicted_df


def _convert_to_embedding_indices(
        unrated_df: pd.DataFrame,
        user_index: pd.Index,
        item_index: pd.Index,
        user_col: str = "userId",
        item_col: str = "movieId"
) -> np.ndarray:
    """
    Converts embeddings to indices

    :param unrated_df:
    :param user_index:
    :param item_index:
    :param user_col:
    :param item_col:
    :return:
    """
    user_map = {user: i for i, user in enumerate(user_index)}
    item_map = {item: i for i, item in enumerate(item_index)}

    try:
        user_ids = unrated_df[user_col].map(user_map)  # type: ignore
        item_ids = unrated_df[item_col].map(item_map)  # type: ignore
    except KeyError as e:
        raise ValueError(f"ID not found in the training index: {e}")

    if user_ids.isnull().any() or item_ids.isnull().any():
        raise ValueError("Some IDs were not found in the training index.")

    return np.stack([user_ids.astype(np.int64), item_ids.astype(np.int64)], axis=1)


def _prepare_prediction_data(unrated_movie_series: pd.Series) -> np.ndarray:
    """
    Extracts features from an unrated movie Series, returns a 2D np array for prediction (shape: [1, num_features])

    :param unrated_movie_series:
    :return: np.ndarray:
    """
    # Convert an unrated movie Series into an 1D numpy ndarray
    # Then converts that 1D ndarray into a 2D ndarray (matrix) (1 row, N columns)
    unrated_movie_array = unrated_movie_series.values.astype(np.float32).reshape(1, -1)
    # Return the 2D array
    return unrated_movie_array


def _predict_rating_for_unrated_movie_with_model_from_ndarray_to_ndarray(input_array: np.ndarray,
                                                                         model: Union[
                                                                             NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
                                                                         model_type: ModelType) -> np.ndarray:
    """
    Predicts the rating of an unrated movie (or a group of unrated movies) using a trained model and an input ndarray

    :param input_array:
    :param model:
    :param model_type:
    :return: np.ndarray:
    """
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            return _predict_rating_for_unrated_movie_with_neural_network_from_ndarray_to_ndarray(input_array,
                                                                                                 model)
        case ModelType.RANDOM_FOREST.value:
            return _predict_rating_for_unrated_movie_with_random_forest_from_ndarray_to_ndarray(input_array,
                                                                                                model)
        case ModelType.XGBOOST.value:
            return _predict_rating_for_unrated_movie_with_xgboost_from_ndarray_to_ndarray(input_array, model)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")



def _predict_rating_for_unrated_movie_with_neural_network_from_ndarray_to_ndarray(
        unrated_movie_array: np.ndarray,
        model: NeuralNetRegressor
) -> np.ndarray:
    """
    Predicts the rating of an unrated movie (or a group of unrated movies) using a nn

    :param unrated_movie_array:
    :param model:
    :return: np.ndarray:
    """
    # Skorch takes care of everything
    prediction = model.predict(unrated_movie_array.astype(np.float32))
    return prediction  # .flatten()


def _predict_rating_for_unrated_movie_with_random_forest_from_ndarray_to_ndarray(unrated_movie_array: np.ndarray,
                                                                                 model: RandomForestRegressor) -> np.ndarray:
    """
    Predicts the rating of an unrated movie (or a group of unrated movies) using a random forest model and an input ndarray

    :param unrated_movie_array:
    :param model:
    :return: np.ndarray:
    """
    prediction = model.predict(unrated_movie_array.astype(np.float32))
    return prediction  # shape: (N,)


def _predict_rating_for_unrated_movie_with_xgboost_from_ndarray_to_ndarray(unrated_movie_array: np.ndarray,
                                                                           model: xgb.XGBRegressor) -> np.ndarray:
    """
    Predicts the rating of an unrated movie (or a group of unrated movies) using a xgboost model and an input ndarray

    :param unrated_movie_array:
    :param model:
    :return: np.ndarray:
    """
    prediction = model.predict(unrated_movie_array.astype(np.float32))
    return prediction  # shape: (N,)
