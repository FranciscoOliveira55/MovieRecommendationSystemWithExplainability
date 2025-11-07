from typing import Tuple, Union

from matplotlib.figure import Figure

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from skorch import NeuralNetRegressor

from configs import ModelType
from core import explainability

if __name__ == '__main__':
    pass


def generate_local_explanation_for_predictions_with_cf(
        selected_user_id: int,
        model: Union[NeuralNetRegressor],  # Trained model.
        rated_df_used_in_training: pd.DataFrame,
        user_index: pd.Index,  # Indexes of users collected in training
        predicted_movie_to_explain: pd.DataFrame,  # Df with only 1 movie (the one to explain)
) -> Tuple[pd.DataFrame, str]:
    """
    Generate local explanations for cf predictions

    :param selected_user_id:
    :param model:
    :param rated_df_used_in_training:
    :param user_index:
    :param predicted_movie_to_explain:
    :return: Tuple[pd.DataFrame, str]:
    """
    ratings_of_similar_users_df = explainability.explain_local_predictions_for_cf(
        selected_user_id=selected_user_id,
        model=model,
        rated_df_used_in_training=rated_df_used_in_training,
        user_index=user_index,
        predicted_movie_to_explain=predicted_movie_to_explain
    )

    explanation_text:str = explainability.generate_neighbour_based_explanation_text(ratings_of_similar_users_df)

    print(ratings_of_similar_users_df)
    return ratings_of_similar_users_df, explanation_text


def generate_shap_explanation_and_filter_out_zero_importance_features(
        prediction_model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        rated_movies_dataset: pd.DataFrame,
        movies_to_explain_df: pd.DataFrame,
        global_or_local: str,
        model_type: ModelType
) -> Tuple[np.ndarray, list[str]]:
    """
    Generate SHAP explanation for a batch of movie(s) (global/local explainability)

    :param prediction_model:
    :param rated_movies_dataset:
    :param movies_to_explain_df:
    :param global_or_local:
    :param model_type:
    :return: Tuple[np.ndarray, list[str]]:
    """
    # Generate SHAP explanation for a batch of movie(s) (global/local explainability)
    if f"shap_values_{global_or_local}_explanation" not in st.session_state:
        shap_values_explanation, expected_value = explainability.explain_predictions_with_shap_for_cbf(
            model=prediction_model,
            rated_movies_dataset=rated_movies_dataset,
            predicted_movies_to_explain=movies_to_explain_df,
            model_type=model_type
        )
        st.session_state[f"shap_values_{global_or_local}_explanation"] = shap_values_explanation
    else:
        shap_values_explanation = st.session_state[f"shap_values_{global_or_local}_explanation"]

    # Filter SHAP values and feature names to exclude near-zero importance features
    feature_columns = movies_to_explain_df.drop(columns=["PredictedRating"]).columns
    # st.write(shap_values_explanation.shape)
    # st.write(shap_values_explanation.ndim)
    # st.write(shap_values_explanation)
    # st.write(feature_columns.tolist())
    # st.write(shap_values_explanation.__class__.__name__)

    filtered_shap_values_explanation, filtered_feature_names_explanation = explainability.filter_zero_shap_features(
        shap_values_explanation, feature_columns.tolist())
    return filtered_shap_values_explanation, filtered_feature_names_explanation



def get_shap_bar_and_pie_chart_figures(filtered_shap_values_explanation: np.ndarray,
                                       filtered_feature_names_explanation: list[str],
                                       global_or_local: str) -> Tuple[Figure, Figure]:
    """
    Generate bar and pie charts figures

    :param filtered_shap_values_explanation:
    :param filtered_feature_names_explanation:
    :param global_or_local:
    :return: Tuple[Figure, Figure]:
    """
    fig_bar = explainability.plot_shap_bar(
        f"Feature Importance ({global_or_local})",
        filtered_shap_values_explanation,
        filtered_feature_names_explanation
    )
    fig_pie = explainability.plot_shap_pie(f"Feature Importance ({global_or_local})",
                                           filtered_shap_values_explanation,
                                           filtered_feature_names_explanation)
    return fig_bar, fig_pie
