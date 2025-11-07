from typing import Union

import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from skorch import NeuralNetRegressor
from core import evaluation, model
from configs import ModelType, FilteringType

if __name__ == '__main__':
    pass


def cross_validate_model_for_evaluation(
        prediction_model: Union[NeuralNetRegressor],  # Trained model being used
        rated_df: pd.DataFrame,
        selected_user: str,
        selected_model: ModelType,  # Type of model being used
        filtering_type:FilteringType
):
    """
    Cross validates the model for evaluation

    :param prediction_model:
    :param rated_df:
    :param selected_user:
    :param selected_model:
    :param filtering_type:
    :return:
    """
    cv_results_session_key = f"cv_results_{selected_model}_{selected_user}_{filtering_type}"
    if cv_results_session_key not in st.session_state:
        # Get input and output features
        x_input_features, y_output_feature, _, _, _, _ = model._prepare_training_data(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=selected_model,
            filtering_type=filtering_type
        )

        cv_results = evaluation.cross_validate_model(
            model=prediction_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            model_type=selected_model,
            filtering_type=filtering_type,
            selected_user_id=selected_user
        )
        st.session_state[cv_results_session_key] = cv_results
    else:
        cv_results = st.session_state[cv_results_session_key]
    return cv_results


def prepare_cross_validation_results_for_display(cv_results: dict):
    """
    Prepares the cross validation results for display

    :param cv_results:
    :return:
    """
    # ----- 1. Table per fold -----
    cv_metrics_per_fold_df = pd.DataFrame({
        "Fold": list(range(1, len(cv_results["test_mse"]) + 1)),
        "Test MSE": -cv_results["test_mse"],
        "Train MSE": -cv_results["train_mse"],
        "Test RMSE": -cv_results["test_rmse"],
        "Train RMSE": -cv_results["train_rmse"],
        "Test MAE": -cv_results["test_mae"],
        "Train MAE": -cv_results["train_mae"],
        "Test R²": cv_results["test_r2"],
        "Train R²": cv_results["train_r2"],
    })

    # ----- 2. Summary Mean ± standard deviation -----
    mean_values = {}
    std_values = {}

    for metric in cv_metrics_per_fold_df.columns[1:]:  # Ignores 'Fold'
        values = cv_metrics_per_fold_df[metric]
        mean_values[metric] = values.mean()
        std_values[metric] = values.std()

    # Df with 2 rows Mean e Std
    summary_mean_and_standard_deviation_df = pd.DataFrame([mean_values, std_values], index=["Mean", "Std"])

    # ----- 3. Comparative chart (Bar Plot or Boxplot) -----
    metrics_to_plot = ["Test MSE", "Train MSE", "Test RMSE", "Train RMSE"]
    cv_performance_per_fold_chart_figure, ax = plt.subplots()
    cv_metrics_per_fold_df[metrics_to_plot].plot(kind='bar', ax=ax)
    plt.xlabel("Fold")
    plt.ylabel("Metric Value")
    plt.title("Cross-Validation Performance per Fold")
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Boxplot for variability
    cv_variance_per_metric_figure, ax2 = plt.subplots()
    cv_metrics_per_fold_df[metrics_to_plot].plot(kind='box', ax=ax2)
    plt.title("Metric Distribution Across Folds")

    return cv_metrics_per_fold_df, summary_mean_and_standard_deviation_df, cv_performance_per_fold_chart_figure, cv_variance_per_metric_figure


def calc_learning_curve_of_model_for_evaluation(
        prediction_model: Union[NeuralNetRegressor],  # Trained model being used
        rated_df: pd.DataFrame,
        selected_user: str,
        selected_model: ModelType,  # Type of model being used
        filtering_type:FilteringType,
        learning_curve_scoring_metric_key: str
) -> dict:
    """
    Calculates the larning curve of model for evaluation

    :param prediction_model:
    :param rated_df:
    :param selected_user:
    :param selected_model:
    :param filtering_type:
    :param learning_curve_scoring_metric_key:
    :return: dict:
    """
    learning_curve_session_key = f"learning_curve_{learning_curve_scoring_metric_key}_{selected_model}_{selected_user}_{filtering_type}"

    if learning_curve_session_key not in st.session_state:
        # Get input and output features
        x_input_features, y_output_feature, _, _, _, _ = model._prepare_training_data(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=selected_model,
            filtering_type=filtering_type
        )


        learning_curve = evaluation.calc_learning_curve_of_model(
            model=prediction_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            learning_curve_scoring_metric_key=learning_curve_scoring_metric_key,
            model_type=selected_model,
            filtering_type=filtering_type,
            selected_user_id=selected_user
        )
        st.session_state[learning_curve_session_key] = learning_curve
    else:
        learning_curve = st.session_state[learning_curve_session_key]
    return learning_curve


def prepare_learning_curve_for_display(learning_curve_dict: dict) -> Figure:
    """
    Prepares the learning curve results for display

    :param learning_curve_dict:
    :return:
    """
    train_sizes = learning_curve_dict['train_sizes']
    train_scores = learning_curve_dict['train_scores']
    test_scores = learning_curve_dict['test_scores']

    # Calculate mean and std deviation for training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    learning_curve_fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training scores with shaded std deviation area
    ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Train score')
    ax.fill_between(train_sizes,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1, color='blue')

    # Plot validation scores with shaded std deviation area
    ax.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validation score')
    ax.fill_between(train_sizes,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1, color='green')

    # Set plot title and labels
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training Set Size')
    learning_curve_scoring_metric_key = st.session_state["learning_curve_scoring_metric_key"]
    ax.set_ylabel(f'Score ({learning_curve_scoring_metric_key})')
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()

    return learning_curve_fig
