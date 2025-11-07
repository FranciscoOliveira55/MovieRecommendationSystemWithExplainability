import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from skorch import NeuralNetRegressor
from torch import nn
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
from typing import Tuple, Union
import matplotlib.pyplot as plt
from configs import ModelType, ShapConfig, ExplainabilityConfigForCF
from core import prediction
from core.utils import write_log

if __name__ == '__main__':
    pass
    print("Hi there, you're in the explainability module :)")
    print(
        "This module uses SHAP explainability tools (KernelExplainer, plots, etc.) to explain movie rating predictions :)")


def explain_local_predictions_for_cf(
        selected_user_id: int,
        model: Union[NeuralNetRegressor],  # Trained model.
        rated_df_used_in_training: pd.DataFrame,
        user_index: pd.Index,  # Indexes of users collected in training
        predicted_movie_to_explain: pd.DataFrame,  # Df with only 1 movie (the one to explain)
) -> pd.DataFrame:
    """
    Generates local neighbour based explanations for cf recommendations

    :param selected_user_id:
    :param model:
    :param rated_df_used_in_training:
    :param user_index:
    :param predicted_movie_to_explain:
    :return: pd.DataFrame:
    """
    if predicted_movie_to_explain.empty or rated_df_used_in_training.empty:
        return pd.DataFrame()
    #Cpu or Gpu
    device = model.device
    # Get nn model
    pytorch_model = model.module_
    # 1. Filter users who rated the recommended movie
    recommended_movie_id: int = predicted_movie_to_explain['movieId'].iloc[0]
    users_who_rated_movie = rated_df_used_in_training[rated_df_used_in_training['movieId'] == recommended_movie_id][
        'userId'].unique()
    # 2. Get embeddings only for those users
    all_user_vectors: np.ndarray = pytorch_model.user_embedding.weight.detach().cpu().numpy()
    user_vectors_of_interest: np.ndarray = all_user_vectors[
        [user_index.get_loc(u) for u in users_who_rated_movie if u in user_index]]
    # 3. Get the selected user's embedding vector
    selected_user_tensor = torch.tensor([selected_user_id]).to(device)
    selected_user_vector: np.ndarray = pytorch_model.user_embedding(selected_user_tensor).detach().cpu().numpy()
    # 4. Compute cosine similarity only with these users
    similarities: np.ndarray = cosine_similarity(selected_user_vector, user_vectors_of_interest)[0]
    # 5. Sort and get top_k similar users
    top_k: int = ExplainabilityConfigForCF.TOP_K_MOST_SIMILAR_USERS
    top_indices: np.ndarray = similarities.argsort()[::-1][:top_k]
    # 6. Map back to the original user IDs
    similar_users_ids: np.ndarray = users_who_rated_movie[top_indices]
    # print(f"similar_users_ids: {similar_users_ids}")
    # 7. Get their similarity scores
    similarity_scores: np.ndarray = similarities[top_indices]
    # print(f"similarity_scores: {similarity_scores}")
    # 8. Get the ratings that these similar users gave to the recommended movie
    ratings_of_similar_users_df: pd.DataFrame = rated_df_used_in_training[
        (rated_df_used_in_training['userId'].isin(similar_users_ids)) &
        (rated_df_used_in_training['movieId'] == recommended_movie_id)
        ][['userId', 'movieId', 'rating']]
    # Create a dictionary mapping userId -> similarity score
    similarity_dict: dict = dict(zip(similar_users_ids, similarity_scores))
    # Add a new column 'similarity_score' by mapping userId through the dictionary
    ratings_of_similar_users_df['similarity_score'] = ratings_of_similar_users_df['userId'].map(similarity_dict)
    # print(f"ratings_of_similar_users:\n {ratings_of_similar_users}")
    # Add weighted score = (rating*similarity_score)
    ratings_of_similar_users_df['weighted_similarity_score'] = (
            ratings_of_similar_users_df['rating'] * ratings_of_similar_users_df['similarity_score']
    )
    # Order by weighted_score
    ratings_of_similar_users_df_sorted = ratings_of_similar_users_df.sort_values(
        by='weighted_similarity_score',
        ascending=False
    )
    # Return df
    return ratings_of_similar_users_df_sorted


def generate_neighbour_based_explanation_text(ratings_of_similar_users_df_sorted: pd.DataFrame) -> str:
    """
    Generates the text strings for neighbour based explanations
    :param ratings_of_similar_users_df_sorted:
    :return: str:
    """

    if ratings_of_similar_users_df_sorted.empty:
        return "ℹ️ No other users rated this movie."

    # Generate explanation message
    ratings_of_similar_users_df_sorted['userId'] = ratings_of_similar_users_df_sorted['userId'].astype(
        int)  # remove casas decimais
    explanation_lines = []
    for _, row in ratings_of_similar_users_df_sorted.iterrows():
        line = (
            f"User 👤 with ID {int(float(row['userId']))} (that has a similarity score of : {row['similarity_score']:.2f} 📊) "
            f"rated this movie {row['rating']:.1f} ⭐."
        )
        explanation_lines.append(line)

    explanation_text = (
            f"ℹ️ We recommended this movie because:\n- " +
            "\n- ".join(explanation_lines)
    )
    # Return the explanation text
    return explanation_text


def explain_local_predictions_for_hyb(
        selected_user_id: int,
        # model: Union[NeuralNetRegressor],  # Trained model.
        rated_uncleaned_df: pd.DataFrame,
        predicted_movie_to_explain: pd.DataFrame,  # Df with only 1 movie (the one to explain)
) -> pd.DataFrame:
    """
    Generates local neighbour based explanations for hybrid recommendations

    :param selected_user_id:
    :param rated_uncleaned_df:
    :param predicted_movie_to_explain:
    :return: pd.DataFrame:
    """
    # 1. Filter users who rated the recommended movie
    recommended_movie_id: int = predicted_movie_to_explain['movieId'].iloc[0]
    write_log(f"recommended_movie_id: {recommended_movie_id}")
    users_who_rated_movie: np.ndarray = rated_uncleaned_df[rated_uncleaned_df['movieId'] == recommended_movie_id][
        'userId'].unique()

    if len(users_who_rated_movie) == 0:
        return pd.DataFrame() # Return empty dataframe
        # raise ValueError("users_who_rated_movie está vazio — nenhum utilizador avaliou este filme.")

    print(f"users_who_rated_movie: {users_who_rated_movie}")
    print(f"rated_uncleaned_df: {rated_uncleaned_df}")

    # 2. Get all user embeddings
    all_user_vectors: pd.DataFrame = rated_uncleaned_df.filter(regex=r"^user", axis=1).drop_duplicates()
    print(f"all_user_vectors: {all_user_vectors}")

    # Get user embeddings of users that rated the movie
    user_vectors_of_interest: pd.DataFrame = all_user_vectors[all_user_vectors["userId"].isin(users_who_rated_movie)]
    print(f"user_vectors_of_interest: {user_vectors_of_interest}")

    # 3. Get the selected user's embedding vector
    selected_user_vector = all_user_vectors[all_user_vectors["userId"] == selected_user_id].iloc[[0]]
    print(f"selected_user_vector: {selected_user_vector}")

    # Set userId as index
    selected_user_vector = selected_user_vector.set_index("userId")
    user_vectors_of_interest = user_vectors_of_interest.set_index("userId")

    # 4. Compute cosine similarity only with these users
    similarities: np.ndarray = cosine_similarity(selected_user_vector, user_vectors_of_interest)[0]
    print(f"similarities: {similarities}")
    # 5. Sort and get top_k similar users
    top_k: int = ExplainabilityConfigForCF.TOP_K_MOST_SIMILAR_USERS
    top_indices: np.ndarray = similarities.argsort()[::-1][:top_k]  # 3,5,1
    print(f"top_indices: {top_indices}")
    # 6. Map back to the original user IDs
    similar_users_ids: np.ndarray = users_who_rated_movie[top_indices]
    print(f"similar_users_ids: {similar_users_ids}")

    # 7. Get their similarity scores
    similarity_scores: np.ndarray = similarities[top_indices]
    print(f"similarity_scores: {similarity_scores}")

    # 8. Get the ratings that these similar users gave to the recommended movie
    ratings_of_similar_users_df: pd.DataFrame = rated_uncleaned_df[
        (rated_uncleaned_df['userId'].isin(similar_users_ids)) &
        (rated_uncleaned_df['movieId'] == recommended_movie_id)
        ][['userId', 'movieId', 'rating']]
    # Create a dictionary mapping userId -> similarity score
    similarity_dict: dict = dict(zip(similar_users_ids, similarity_scores))
    # Add a new column 'similarity_score' by mapping userId through the dictionary
    ratings_of_similar_users_df['similarity_score'] = ratings_of_similar_users_df['userId'].map(similarity_dict)
    # print(f"ratings_of_similar_users:\n {ratings_of_similar_users}")
    # Add weighted score = (rating*similarity_score)
    ratings_of_similar_users_df['weighted_similarity_score'] = (
            ratings_of_similar_users_df['rating'] * ratings_of_similar_users_df['similarity_score']
    )
    # Order by weighted_score
    ratings_of_similar_users_df_sorted = ratings_of_similar_users_df.sort_values(
        by='weighted_similarity_score',
        ascending=False
    )

    # Return df
    return ratings_of_similar_users_df_sorted


def explain_predictions_with_shap_for_cbf(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],  # Trained model.
        rated_movies_dataset: pd.DataFrame,  # Numpy array used as reference background for SHAP.
        predicted_movies_to_explain: pd.DataFrame,
        # Pandas Dataframe containing features of one or more movies (including "PredictedRating" column).
        model_type: ModelType
) -> Tuple[np.ndarray, float]:
    """
    Explain the contribution of each feature for the prediction of either a single movie (local explainability) or a batch of movies (global explainability) using SHAP KernelExplainer.
    Returns: shap_values: Array of SHAP values for each feature, expected_value: Base value from SHAP explanation (expected model output).

    :param model:
    :param rated_movies_dataset:
    :param predicted_movies_to_explain:
    :param model_type:
    :return: Tuple[np.ndarray, float]:
    """
    # Prepare background_data for shap
    background_data = _prepare_background_data(
        rated_movies_dataset=rated_movies_dataset,
        target_column="rating",
    )
    write_log(
        f"Explaining predictions, model_type:{model_type}, background_data_size:{background_data.shape[0]}, batch_to_explain_size:{predicted_movies_to_explain.shape[0]} ... ")
    # Gets the input features from the movies (only dependent variables, not "PredictedRating" column)
    input_features = predicted_movies_to_explain.drop(columns=['PredictedRating']).values.astype(np.float32)

    # Generate shap values to explain the predicted movies
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            shap_values, expected_value = _explain_predictions_with_deep_explainer(model, background_data,
                                                                                   input_features)
        case ModelType.RANDOM_FOREST.value | ModelType.XGBOOST.value:
            shap_values, expected_value = _explain_predictions_with_tree_explainer(model, background_data,
                                                                                   input_features)
        case "agnostic":
            shap_values, expected_value = _explain_predictions_with_kernel_explainer(model, background_data,
                                                                                     input_features, model_type)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")
    # Shap_values is a 2D/3D array (number_of_movies, number_of_features)
    # Lets consolidate it in a 1D array, with the means of each column
    shap_values = np.mean(shap_values, axis=0)  # shape: (n_features,)
    write_log(f"Explaining predictions completed successfully")
    # Return shap_values and expected_value
    return shap_values, expected_value


def _prepare_background_data(
        rated_movies_dataset: pd.DataFrame,  # DataFrame with movies and their features + ratings.
        target_column: str = 'rating',  # The name of the target column to exclude (e.g., 'rating').
) -> np.ndarray:  # Numpy array of shape (sample_size, n_features) containing background feature data.
    """
    Prepares a sample of rated movies as background data for SHAP to compare (without rating column or any column label, just the values)

    :param rated_movies_dataset:
    :param target_column:
    :return: np.ndarray:
    """
    write_log(f"Preparing explaining background_data ... ")
    # Removes target column (rating)
    rated_movies_dataset = rated_movies_dataset.drop(columns=[target_column])
    # Make sure sample_size does not exceed available rows
    sample_size = min(ShapConfig.BACKGROUND_DATA_SAMPLE_SIZE, len(rated_movies_dataset))
    # Sample random subset to use as background data
    background_df = rated_movies_dataset.sample(sample_size, random_state=ShapConfig.BACKGROUND_DATA_RANDOM_STATE)
    # Convert to numpy array for SHAP
    background_data = background_df.values.astype(np.float32)
    write_log(f"Preparing explaining background_data completed")
    return background_data


def _explain_predictions_with_kernel_explainer(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        background_data: np.ndarray,
        input_features: np.ndarray,
        model_type: ModelType
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates shap data with Agnostic explainer, works with every model, but its slower and less precise

    :param model:
    :param background_data:
    :param input_features:
    :param model_type:
    :return: Tuple[np.ndarray, np.ndarray]:
    """

    # Define a wrapper prediction function for SHAP that accepts numpy arrays
    def model_predict(unrated_movie_array: np.ndarray) -> np.ndarray:
        # Call the prediction function and give it the unrated movie feature's values array as well as the AI model
        # Is calling "_predict_rating_for_unrated_movie_with_model_from_ndarray_to_ndarray" from prediction module
        return prediction._predict_rating_for_unrated_movie_with_model_from_ndarray_to_ndarray(unrated_movie_array,
                                                                                               model, model_type)

    # Create a SHAP explainer with background data and model prediction function
    explainer = shap.KernelExplainer(model_predict, data=background_data)
    # Compute SHAP values for the single input (returns list with one array, shape (1, num_features))
    shap_values = explainer.shap_values(input_features)
    # Extract expected_value (base value) from explainer (scalar)
    expected_value = explainer.expected_value
    # shap_values is a list of arrays (for regression usually one element), take first element and flatten
    shap_values = shap_values[0]
    # Return shap_values and expected_value
    return shap_values, expected_value


def _explain_predictions_with_deep_explainer(
        model: NeuralNetRegressor,
        background_data: np.ndarray,
        input_features: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Generates shap data with Deep explainer, works with torch neural networks

    :param model:
    :param background_data:
    :param input_features:
    :return: Tuple[np.ndarray, float]:
    """
    #cpu or gpu
    device = model.device

    # Deep explainer needs tensors to work
    background_data_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)
    input_features_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    # Create explainer
    explainer = shap.DeepExplainer(model.module_, data=background_data_tensor)
    # Compute SHAP values for the single input (returns list with one array, shape (1, num_features))
    shap_values_tensor = explainer.shap_values(input_features_tensor)
    # If it returns a list (1 position for each dependent variable, pick the first position (regression)),
    shap_values = shap_values_tensor  # shap_values_tensor[0] if isinstance(shap_values_tensor, list) else shap_values_tensor
    # remove 3rd dimension (movie, feature, n_dependent_variables (regression))
    # shap_values = shap_values.squeeze()
    # Extract expected value
    expected_value = explainer.expected_value
    # If it returns a list (1 position for each dependent variable, pick the first position (regression)),
    expected_value = expected_value[0] if isinstance(expected_value, list) else expected_value
    # Make sure everything is ndarray
    shap_values = shap_values.cpu().numpy() if isinstance(shap_values, torch.Tensor) else shap_values
    return shap_values, float(expected_value)


def _explain_predictions_with_tree_explainer(
        model: Union[RandomForestRegressor, xgb.XGBRegressor],
        background_data: np.ndarray,
        input_features: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Generates shap data with Tree explainer, works with random forest and xgboost

    :param model:
    :param background_data:
    :param input_features:
    :return: Tuple[np.ndarray, float]:
    """
    explainer = shap.TreeExplainer(model, data=background_data)
    # Compute SHAP values for the single input (returns list with one array, shape (1, num_features))
    shap_values = explainer.shap_values(input_features)
    # Extract expected_value (base value) from explainer (scalar)
    expected_value = explainer.expected_value
    # Return shap_values and expected_value
    return shap_values, float(expected_value)


def plot_shap_bar(
        plot_title: str,
        shap_values: np.ndarray,  # 1D array containing SHAP values for a single example
        feature_names: list[str],
        threshold_pct: float = 1  # only shows contribution over X%
):
    """
    Plots a bar chart of SHAP values for a single prediction. (the recommended movie)

    :param plot_title:
    :param shap_values:
    :param feature_names:
    :param threshold_pct:
    :return: Figure
    """
    abs_values = np.abs(shap_values)
    total = abs_values.sum()

    # mask of relevant features
    mask = (abs_values / total * 100) >= threshold_pct

    filtered_values = shap_values[mask]
    filtered_names = [f for i, f in enumerate(feature_names) if mask[i]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(filtered_names, filtered_values)
    ax.set_title(plot_title)

    plt.yticks(fontsize=12)
    plt.tight_layout()

    return fig


def plot_shap_pie(
        plot_title: str,
        shap_values: np.ndarray,
        feature_names: list[str],
        threshold_pct: float = 2
):
    """
    Plots a pie chart of the SHAP values for a single prediction. (the recommended movie)

    :param plot_title:
    :param shap_values:
    :param feature_names:
    :param threshold_pct:
    :return: Figure
    """
    abs_values = np.abs(shap_values)
    total = abs_values.sum()

    # Determines what features are too small
    small_mask = (abs_values / total * 100) < threshold_pct
    large_values = abs_values[~small_mask]
    large_labels = [f for i, f in enumerate(feature_names) if not small_mask[i]]

    # Groups small slices in 'Other'
    if small_mask.any():
        large_values = np.append(large_values, abs_values[small_mask].sum())
        large_labels.append('Other')

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        large_values,
        labels=large_labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct >= threshold_pct else '',
        startangle=140,
        pctdistance=0.85,
        labeldistance=1.05
    )

    # Adjusts font size
    for t in texts + autotexts:
        t.set_fontsize(12)

    for w, t in zip(wedges, texts):
        ang = (w.theta2 + w.theta1) / 2

        if ang > 90 and ang < 270:
            t.set_rotation(ang + 180)
            t.set_horizontalalignment('right')
        else:
            t.set_rotation(ang)
            t.set_horizontalalignment('left')

        t.set_verticalalignment('center')
        t.set_rotation_mode("anchor")

    ax.set_title(plot_title, pad=50)
    ax.axis('equal')
    plt.tight_layout()
    return fig


def filter_zero_shap_features(
        shap_values: np.ndarray,  # 1D numpy array of SHAP values for each feature.
        feature_names: list[str],  # List of feature names corresponding to shap_values.
        threshold: float = 1e-8  # Threshold below which SHAP values are considered zero (default 1e-8).
) -> tuple[np.ndarray, list[str]]:
    """
    # Filter out features whose absolute SHAP values are below or equal to the threshold (effectively zero).

    :param shap_values:
    :param feature_names:
    :param threshold:
    :return: tuple[np.ndarray, list[str]]:
    """
    # Calculate the absolute values of SHAP values to consider magnitude only (ignore sign)
    abs_shap = np.abs(shap_values)
    # Create a boolean mask selecting SHAP values with magnitude greater than the threshold
    mask = abs_shap > threshold
    # Filter the SHAP values using the mask to keep only significant values
    filtered_shap_values = shap_values[mask]
    # Filter feature names to keep only those corresponding to significant SHAP values
    filtered_feature_names = [feature_names[i] for i, keep in enumerate(mask) if keep]
    # Return the filtered SHAP values and filtered features names
    return filtered_shap_values, filtered_feature_names
