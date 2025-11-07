import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, learning_curve
from skorch import NeuralNetRegressor
import xgboost as xgb

from core.utils import write_log
from configs import ModelType, FilePaths, FilteringType

from configs import CrossValidationConfig, LEARNINGCURVEConfig

if __name__ == '__main__':
    pass
    print("Hi there, you're in the evaluation module :)")
    print("This module evaluates an AI model using cross-validation and metrics :)")


def cross_validate_model(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        filtering_type: FilteringType,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
        model_type: ModelType,
        selected_user_id: str
) -> dict:
    """
    Cross validates the model for evaluation

    :param model:
    :param filtering_type:
    :param x_input_features:
    :param y_output_feature:
    :param model_type:
    :param selected_user_id:
    :return: dict:
    """
    # Tries to read cross validate data, if it doesn't exist, then calculates new one
    cv_results: dict
    try:
        write_log("Loading cross validation data ...")
        cv_results = _read_cross_validation_or_learning_curve_locally(
            model_type=model_type,
            filtering_type=filtering_type,
            selected_user=selected_user_id,
            cross_validation_or_learning_curve="cross_validation")
        write_log(f"Successfully loaded cross validation data, model_type:{model_type}, user_id:{selected_user_id}")
    except FileNotFoundError:
        write_log("Validation data not found, calculating new one ...")
        cv_results = cross_validate(
            model,
            x_input_features.astype(np.float32),
            y_output_feature.astype(np.float32),
            cv=CrossValidationConfig.CV_FOLDS,
            scoring=CrossValidationConfig.SCORING,
            return_train_score=CrossValidationConfig.RETURN_TRAIN_SCORE,
            return_estimator=CrossValidationConfig.RETURN_ESTIMATORS,  # Return temporary trained models for each fold,
            n_jobs=1,#CrossValidationConfig.N_JOBS,
            verbose=CrossValidationConfig.VERBOSE
        )
        # Drop estimator models
        cv_results.pop("estimator", None)
        # Normalize metric values
        # I'm doing that on app_evaluation ... need update
        # Save data locally
        _save_cross_validation_or_learning_curve_result_locally(
            cv_results_or_learning_curve_dict=cv_results,
            model_type=model_type,
            filtering_type=filtering_type,
            selected_user=selected_user_id,
            cross_validation_or_learning_curve="cross_validation")
        write_log(
            f"Successfully calculated and saved cross validation data, model_type:{model_type}, user_id:{selected_user_id}")

    # Trained temporary models
    # trained_models = cv_results['estimator']

    return cv_results


def calc_learning_curve_of_model(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        filtering_type: FilteringType,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
        learning_curve_scoring_metric_key: str,
        model_type: ModelType,
        selected_user_id: str
) -> dict:
    """
    Calculates learning curve for model evaluation

    :param model:
    :param filtering_type:
    :param x_input_features:
    :param y_output_feature:
    :param learning_curve_scoring_metric_key:
    :param model_type:
    :param selected_user_id:
    :return: dict:
    """
    learning_curve_dict: dict
    try:
        write_log("Loading learning curve data ...")
        learning_curve_dict = _read_cross_validation_or_learning_curve_locally(
            model_type=model_type,
            filtering_type=filtering_type,
            selected_user=selected_user_id,
            cross_validation_or_learning_curve="learning_curve",
            learning_curve_scoring_metric_key=learning_curve_scoring_metric_key
        )
        write_log(
            f"Successfully loaded learning curve data, model_type:{model_type}, user_id:{selected_user_id}, learning_curve_scoring_metric_key:{learning_curve_scoring_metric_key}")
    except FileNotFoundError:
        write_log("Learning curve data not found, calculating new one ...")
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            x_input_features.astype(np.float32),
            y_output_feature.astype(np.float32),
            cv=LEARNINGCURVEConfig.CV_FOLDS,
            scoring=LEARNINGCURVEConfig.SCORING_METRICS[learning_curve_scoring_metric_key],
            train_sizes=LEARNINGCURVEConfig.TRAIN_SIZES,
            n_jobs=LEARNINGCURVEConfig.N_JOBS,
            shuffle=LEARNINGCURVEConfig.SHUFFLE,
            random_state=LEARNINGCURVEConfig.RANDOM_STATE,
            verbose=LEARNINGCURVEConfig.VERBOSE
        )

        # Unless it's r2, invert signal of scores
        if learning_curve_scoring_metric_key != "r2":
            train_scores = -train_scores
            test_scores = -test_scores
        # Turn in dictionary
        learning_curve_dict = {
            "train_sizes": train_sizes,
            "train_scores": train_scores,
            "test_scores": test_scores
        }
        # Save data locally
        _save_cross_validation_or_learning_curve_result_locally(
            cv_results_or_learning_curve_dict=learning_curve_dict,
            model_type=model_type,
            filtering_type=filtering_type,
            selected_user=selected_user_id,
            cross_validation_or_learning_curve="learning_curve",
            learning_curve_scoring_metric_key=learning_curve_scoring_metric_key
        )
        write_log(
            f"Successfully calculated and saved learning curve data, model_type:{model_type}, user_id:{selected_user_id}, using metric: {learning_curve_scoring_metric_key}")

    return learning_curve_dict


def _save_cross_validation_or_learning_curve_result_locally(
        cv_results_or_learning_curve_dict: dict,
        model_type: ModelType,
        filtering_type: FilteringType,
        selected_user: str,
        cross_validation_or_learning_curve: str = "cross_validation",
        learning_curve_scoring_metric_key: str = None
):
    """
    Stores the cross validation or learning curve results in json files.

    :param cv_results_or_learning_curve_dict:
    :param model_type:
    :param filtering_type:
    :param selected_user:
    :param cross_validation_or_learning_curve:
    :param learning_curve_scoring_metric_key:
    :return:
    """
    # Gets data path
    file_path: str = FilePaths.get_crossed_validation_or_learning_curve_path(
        model_type=model_type,
        user_id=int(float(selected_user)),
        cross_validation_or_learning_curve=cross_validation_or_learning_curve,
        filtering_type=filtering_type
    )
    # Add learning_curve_scoring_metric_key if it exists
    learning_curve_scoring_metric_key_suffix = f"_{learning_curve_scoring_metric_key}" if learning_curve_scoring_metric_key is not None else ""
    file_path_complete: Path = Path(f"{file_path}{learning_curve_scoring_metric_key_suffix}.pkl")
    # Save using pickle
    with open(file_path_complete, "wb") as f:
        pickle.dump(cv_results_or_learning_curve_dict, f)  # type: ignore


def _read_cross_validation_or_learning_curve_locally(
        model_type: ModelType,
        filtering_type: FilteringType,
        selected_user: str,
        cross_validation_or_learning_curve: str = "cross_validation",
        learning_curve_scoring_metric_key: str = None
) -> dict:
    """
    Reads cross validation or learning curve data from json files

    :param model_type:
    :param filtering_type:
    :param selected_user:
    :param cross_validation_or_learning_curve:
    :param learning_curve_scoring_metric_key:
    :return: dict:
    """
    # Gets the expected file path
    file_path: str = FilePaths.get_crossed_validation_or_learning_curve_path(
        model_type=model_type,
        user_id=int(float(selected_user)),
        cross_validation_or_learning_curve=cross_validation_or_learning_curve,
        filtering_type=filtering_type
    )
    # Add learning_curve_scoring_metric_key if it exists
    learning_curve_scoring_metric_key_suffix = f"_{learning_curve_scoring_metric_key}" if learning_curve_scoring_metric_key is not None else ""
    file_path_complete: Path = Path(f"{file_path}{learning_curve_scoring_metric_key_suffix}.pkl")
    if not file_path_complete.exists():
        raise FileNotFoundError()
    # Load using pickle
    with open(file_path_complete, "rb") as f:
        return pickle.load(f)
