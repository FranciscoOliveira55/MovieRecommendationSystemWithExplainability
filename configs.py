from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

"""
This is a configuration file
"""


class ModelType(Enum):
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "extreme_gradient_boosting"


class FilteringType(Enum):
    CONTENT_BASED_FILTERING = "content_based_filtering"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    HYBRID_FILTERING = "hybrid_filtering"


class FilePaths:
    _BASE_DIR: Path = Path(__file__).resolve().parent

    # Datasets directories path
    _DATASETS_DIR_PATH: Path = Path(_BASE_DIR / "datasets")
    DATASETS_BASE_DIR_PATH: Path = Path(_DATASETS_DIR_PATH / "base")
    DATASETS_ADDED_DIR_PATH: Path = Path(_DATASETS_DIR_PATH / "added")
    DATASETS_TRANSFORMED_DIR_PATH: Path = Path(_DATASETS_DIR_PATH / "transformed")

    # Base datasets path
    MOVIES_CSV_PATH: str = str(DATASETS_BASE_DIR_PATH / "movie.csv")
    RATINGS_CSV_PATH: str = str(DATASETS_BASE_DIR_PATH / "rating.csv")
    LINKS_CSV_PATH: str = str(DATASETS_BASE_DIR_PATH / "link.csv")

    # Added users, items and interactions paths
    ADDED_MOVIES_CSV_PATH: str = str(DATASETS_ADDED_DIR_PATH / "added_movies.csv")
    ADDED_MOVIES_TMDB_DETAILS_CSV_PATH: str = str(DATASETS_ADDED_DIR_PATH / "added_movies_tmdb_details.csv")
    ADDED_USERS_CSV_PATH: str = str(DATASETS_ADDED_DIR_PATH / "added_users.csv")
    ADDED_RATINGS_CSV_PATH: str = str(DATASETS_ADDED_DIR_PATH / "added_ratings.csv")

    # Logs path
    RUNNING_LOGS_PATH: str = str(_BASE_DIR / "logs.txt")

    @staticmethod
    def get_transformed_df_path(
            filtering_type: FilteringType,
            selected_user_id: int,
            clean_and_index_dfs: bool
    ) -> Tuple[str, str]:
        # Get df path
        match filtering_type.value:
            case FilteringType.CONTENT_BASED_FILTERING.value:
                transformed_rated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"ratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
                transformed_unrated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"unratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
            case FilteringType.COLLABORATIVE_FILTERING.value:
                transformed_rated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"ratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
                transformed_unrated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"unratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
            case FilteringType.HYBRID_FILTERING.value:
                transformed_rated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"ratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatingsCbf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSampleCbf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_pickUsersWithMostRatingsCf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSampleCf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
                transformed_unrated_movies_df_path: Path = FilePaths.DATASETS_TRANSFORMED_DIR_PATH / filtering_type.value / f"unratedMoviesDf_selectedUserId{selected_user_id}_pickUsersWithMostRatingsCbf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSampleCbf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_pickUsersWithMostRatingsCf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSampleCf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}_clean_and_index_dfs_{str(clean_and_index_dfs).lower()}.csv"
            case _:
                raise ValueError(f"Unsupported filtering type: {filtering_type}, type={type(filtering_type)}")

        # Make parent dir if it doesn't exist
        transformed_rated_movies_df_path.parent.mkdir(parents=True, exist_ok=True)
        # Return model path
        return str(transformed_rated_movies_df_path), str(transformed_unrated_movies_df_path)

    _SAVED_MODELS_DIR_PATH: Path = Path(_BASE_DIR / "saved_models")

    @staticmethod
    def get_model_path(
            model_type: ModelType,
            filtering_type: FilteringType,
            selected_user_id: int
    ) -> str:

        # Get model path
        match filtering_type.value:
            case FilteringType.CONTENT_BASED_FILTERING.value:
                model_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userId{str(selected_user_id)}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}"
            case FilteringType.COLLABORATIVE_FILTERING.value:
                model_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userId{str(selected_user_id)}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}"
            case FilteringType.HYBRID_FILTERING.value:
                model_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userIdXXX_pickUsersWithMostRatingsCbf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSampleCbf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_pickUsersWithMostRatingsCf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSampleCf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}"
            case _:
                raise ValueError(f"Unsupported filtering type: {filtering_type}, type={type(filtering_type)}")

        # Create parents directory if needed
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # Return model path
        return str(model_path)

    @staticmethod
    def get_crossed_validation_or_learning_curve_path(
            model_type: ModelType,
            user_id: int,
            cross_validation_or_learning_curve: str,
            filtering_type: FilteringType
    ) -> str:
        # Get path
        match filtering_type.value:
            case FilteringType.CONTENT_BASED_FILTERING.value:
                cv_or_lc_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userId{str(user_id)}_{cross_validation_or_learning_curve}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}"
            case FilteringType.COLLABORATIVE_FILTERING.value:
                cv_or_lc_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userId{str(user_id)}_{cross_validation_or_learning_curve}_pickUsersWithMostRatings_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSample_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}"
            case FilteringType.HYBRID_FILTERING.value:
                cv_or_lc_path: Path = FilePaths._SAVED_MODELS_DIR_PATH / filtering_type.value / model_type.value / f"userId{str(user_id)}_{cross_validation_or_learning_curve}_pickUsersWithMostRatingsCbf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF).lower()}_ratedDfNumberOfXTopUsersSampleCbf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF}_pickUsersWithMostRatingsCf_{str(UserPickConfig.PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF).lower()}_ratedDfNumberOfXTopUsersSampleCf_{UserPickConfig.NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF}"
            case _:
                raise ValueError(f"Unsupported filtering type: {filtering_type}, type={type(filtering_type)}")

        # Create parents directory if needed
        cv_or_lc_path.parent.mkdir(parents=True, exist_ok=True)
        # Return path
        return str(cv_or_lc_path)


class UserPickConfig:
    PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CBF: bool = True
    NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CBF: int = 5
    PICK_USERS_WITH_THE_MOST_RATINGS_FOR_CF: bool = True
    NUMBER_OF_X_TOP_USERS_FOR_RATING_DF_FOR_CF: int = 100


class GridSearchConfig:
    USE_GRID_SEARCH: bool = False


class NeuralNetworkConfigForCBF:
    HIDDEN_DIM: int = [128, 128]  # 16,32,64
    LEARNING_RATE: float = 0.1  # standard between(0.001, 0.01)
    NUM_EPOCHS: int = 70  # 100, 250, 500
    BATCH_SIZE: int = 64  # 64
    ITERATOR_TRAIN_SHUFFLE: bool = True
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    VERBOSE: int = 2
    DROPOUT: float = 0.0

class NeuralNetworkGridSearchConfigForCBF:
    HIDDEN_DIM: list[list[int]] = [[64, 64], [128, 128] ] # 16, 32, 64, 128, 256, [16, 16], [32, 32],
    LEARNING_RATE: list[int] = [0.1, 0.01]  # 0.0005, [0.1, 0.01, 0.001, 0.0001]
    MAX_EPOCHS: list[int] = [70]  # [100, 200, 300, 500]
    BATCH_SIZE: list[int] = [32, 64]  # 64 [16, 32, 64, 128, 256]
    CV: int = 3  # 10
    SCORING: str = "r2" # "neg_mean_squared_error"  # "r2"
    VERBOSE: int = 2
    DROPOUT= [0.0, 0.1]
    N_JOBS: int = 8


class NeuralNetworkConfigForCF:
    HIDDEN_DIM: int = [32, 32]  # 16,32,64
    LEARNING_RATE: float = 0.0005  # standard between(0.001, 0.01)
    NUM_EPOCHS: int = 70  # 100, 250, 500
    BATCH_SIZE: int = 256
    ITERATOR_TRAIN_SHUFFLE: bool = True
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    VERBOSE: int = 2
    EMBEDDING_DIM:int = 64  # 64
    DROPOUT:float = 0.5


class NeuralNetworkGridSearchConfigForCF:
    HIDDEN_DIM: list[list[int]] = [[32, 32]]  # , [64, 46], [128, 128] , [16, 32, 64, 128, 256]
    LEARNING_RATE: list[int] = [0.0005]  # [0.1, 0.01, 0.001, 0.0001]
    MAX_EPOCHS: list[int] = [70]  # [100, 200, 300, 500]
    BATCH_SIZE: list[int] = [256]  # [16, 32, 64, 128, 256]
    CV: int = 3  # 10
    SCORING: str = "r2" # "neg_mean_squared_error"  # "r2"
    VERBOSE: int = 2
    EMBEDDING_DIM: int = [64]  # 64, 128,
    DROPOUT= [0.4, 0.5]
    N_JOBS: int = 8

class RandomForestConfig:
    N_ESTIMATORS: int = 100  # 100, 200, 300
    MAX_DEPTH: int = 10  # 10, None
    RANDOM_STATE: int = 42
    VERBOSE: int = 2
    N_JOBS: int = 8


class XGBoostConfig:
    N_ESTIMATORS: int = 100  # 100, 200, 300
    MAX_DEPTH: int = 10  # 6, 100
    RANDOM_STATE: int = 42
    LEARNING_RATE: float = 0.1  # 0.3, 0.2, 0.1
    OBJECTIVE: str = "reg:squarederror"
    TREE_METHOD: str = 'gpu_hist' if torch.cuda.is_available() else 'hist'  # uses GPU if available
    VERBOSITY: int = 2  # 0 Silence, 1 Errors, 2 Errors,Warnings, 3 Errors, Warnings, Info
    N_JOBS: int = 8


class CrossValidationConfig:
    SCORING: dict = {
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
    }
    CV_FOLDS = KFold(n_splits=3, shuffle=True, random_state=42)  # 5
    RETURN_TRAIN_SCORE: bool = True
    RETURN_ESTIMATORS: bool = True
    #N_JOBS: int = 8
    VERBOSE: int = 1


class LEARNINGCURVEConfig:
    SCORING_METRICS: dict = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
    }
    CV_FOLDS: int = 3
    TRAIN_SIZES: np.ndarray = np.linspace(0.1, 1.0, 5)
    N_JOBS: int = 8  # -1
    SHUFFLE: bool = True
    RANDOM_STATE: int = 42
    VERBOSE: int = 1


class PredictionForRecommendationConfig:
    USE_SAMPLE_INSTEAD_OF_FULL_UNRATED_DF: bool = False  # Full unrated df can be really large
    UNRATED_DF_SAMPLE_SIZE: int = 1000  # Ignored if USE_SAMPLE_INSTEAD_OF_FULL_UNRATED_DF is false
    UNRATED_DF_SAMPLE_RANDOM_STATE: int = 42


class ShapConfig:
    BACKGROUND_DATA_SAMPLE_SIZE: int = 100  # 100 # Number of samples to use as background (default=100).
    BACKGROUND_DATA_RANDOM_STATE: int = 42
    GLOBAL_EXPLANATION_BATCH_SIZE: int = 30  # Number of movies to use for global explanation
    GLOBAL_EXPLANATION_BATCH_RANDOM_STATE: int = 42


class ExplainabilityConfigForCF:
    TOP_K_MOST_SIMILAR_USERS: int = 5


class GetMoviesDetailsFromTmdbApiConfig:
    REQUEST_MULTITHREADING_MAX_WORKERS: int = 8
