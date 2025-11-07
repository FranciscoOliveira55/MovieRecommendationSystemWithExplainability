from pathlib import Path
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
import json
from configs import ModelType, FilePaths, NeuralNetworkConfigForCBF, NeuralNetworkConfigForCF, RandomForestConfig, \
    XGBoostConfig, \
    NeuralNetworkGridSearchConfigForCBF, GridSearchConfig, FilteringType, NeuralNetworkGridSearchConfigForCF
from core.utils import write_log
from skorch import NeuralNetRegressor
from core.utils import filter_json_serializable

if __name__ == '__main__':
    pass
    print("Hi there, you're in the model module :)")
    print("This module defines, trains, and returns an AI model :)")


def read_or_create_model(
        rated_movies_df: pd.DataFrame,
        target_column: str = "rating",
        model_type: ModelType = ModelType.NEURAL_NETWORK,
        filtering_type: FilteringType = FilteringType.CONTENT_BASED_FILTERING,
        user_id: int = 1,
) -> Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], pd.Index, pd.Index]:
    """
    Builds, trains and returns a recommendation AI model on the given dataframe.

    :param rated_movies_df:
    :param target_column:
    :param model_type:
    :param filtering_type:
    :param user_id:
    :return: Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], pd.Index, pd.Index]:
    """
    # Extracts features and targets from the dataframe, as well as input/output dimensions.
    x_input_features: Union[torch.Tensor, np.ndarray]
    y_output_feature: Union[torch.Tensor, np.ndarray]
    input_dimension: Union[int, None]
    output_dimension: Union[int, None]
    x_input_features, y_output_feature, input_dimension_or_num_users, output_dimension_or_num_items, user_index, item_index = _prepare_training_data(
        rated_movies_df=rated_movies_df,
        target_column=target_column,
        model_type=model_type,
        filtering_type=filtering_type
    )
    # Builds the model (empty)
    model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor] = _build_model(
        input_dimension_or_num_users,
        output_dimension_or_num_items,
        model_type, filtering_type)

    # Tries to read the model, if it doesn't exist, then trains a new one
    trained_model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]
    try:
        write_log("Loading model ...")
        trained_model = _read_model_locally(model_type, filtering_type, user_id, input_dimension_or_num_users,
                                            output_dimension_or_num_items)
        write_log(f"Successfully loaded model, model_type:{model_type}, user_id:{user_id}")
    except FileNotFoundError:
        write_log("Model not found, training a new one ...")
        trained_model, gs_results_df = _train_model(model, x_input_features, y_output_feature, model_type,
                                                    filtering_type,
                                                    input_dimension_or_num_users, output_dimension_or_num_items)
        _save_model_locally(trained_model, model_type, filtering_type, user_id)
        if GridSearchConfig.USE_GRID_SEARCH:
            _save_gs_results_locally(gs_results_df, model_type, filtering_type, user_id)
        write_log(f"Successfully trained model, model_type:{model_type}, user_id:{user_id}")

    # Return trained model
    return trained_model, user_index, item_index


def _prepare_training_data(
        rated_movies_df: pd.DataFrame,
        target_column: str,
        model_type: ModelType = ModelType.NEURAL_NETWORK,
        filtering_type: FilteringType = FilteringType.CONTENT_BASED_FILTERING
):
    """
    Prepares training data to train the model

    :param rated_movies_df:
    :param target_column:
    :param model_type:
    :param filtering_type:
    :return:
    """
    match filtering_type.value:
        case FilteringType.CONTENT_BASED_FILTERING.value | FilteringType.HYBRID_FILTERING.value:
            x_input_features, y_output_feature, input_dimension, output_dimension = _prepare_content_filtering_training_data(
                dataset=rated_movies_df,
                target_column=target_column,
                model_type=model_type
            )
            return x_input_features, y_output_feature, input_dimension, output_dimension, None, None
        case FilteringType.COLLABORATIVE_FILTERING.value:
            x_input_features, y_output_features, num_users, num_items, user_index, item_index = _prepare_collaborative_filtering_training_data(
                dataset=rated_movies_df,
                user_col="userId",
                item_col="movieId",
                target_col="rating"
            )
            return x_input_features, y_output_features, num_users, num_items, user_index, item_index
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")



def _prepare_content_filtering_training_data(
        dataset: pd.DataFrame,
        target_column: str = "rating",
        model_type: ModelType = ModelType.NEURAL_NETWORK
) -> Union[Tuple[torch.Tensor, torch.Tensor, int, int], Tuple[np.ndarray, np.ndarray, None, None], Tuple[
    np.ndarray, np.ndarray, int, int]]:
    """
    Extracts features and targets from a DataFrame
    Returns PyTorch tensors (if it's for a nn) and input/output dimensions or ndarray (if it's for random forest or xgboost).

    :param dataset:
    :param target_column:
    :param model_type:
    :return: Union[Tuple[torch.Tensor, torch.Tensor, int, int], Tuple[np.ndarray, np.ndarray, None, None], Tuple[
    np.ndarray, np.ndarray, int, int]]:
    """
    # Drop rows with missing target values (movies without rating)
    df = dataset.dropna(subset=[target_column])
    # Selects the values from the independent variables (e.g., genres+)
    # Returns a numpy ndarray (N dimensional array) (2D) (matrix) with shape (number_of_rows, number_of_columns_after_drop)
    # x[0] — returns row 0 (the first row) as a 1D array (e.g., [1.0, 2.0, 3.0])
    # x[0, 0] — returns the element at row 0, column 0 (e.g., a float like 2.0)
    # x[:, 0] — returns all rows in column 0 as a 1D array (e.g., [0.5, 2.0, 3.4, ...])
    x_input_features = df.drop(columns=[target_column]).values.astype(np.float32)
    # Selects the values from the dependent variable (e.g., rating)
    # Returns a 1D numpy ndarray with shape (number_of_rows)
    y_output_feature = df[target_column].values.astype(np.float32)
    if len(y_output_feature.shape) == 1:
        y_output_feature = y_output_feature.reshape(-1, 1)
    # If it's random forest or xgboost, we can return already
    if model_type in (ModelType.RANDOM_FOREST, ModelType.XGBOOST):
        return x_input_features, y_output_feature, None, None
    # Else convert to PyTorch tensors
    x_input_features_tensor = torch.from_numpy(x_input_features)
    # Convert 1D numpy ndarray to 2D (matrix with only 1 column)
    # (x.unsqueeze(0).shape)  # Output: torch.Size([1, 3])   (1 row, N columns)
    # (x.unsqueeze(1).shape)  # Output: torch.Size([3, 1])   (N rows, 1 column)
    # y[0] = 4.5 -> y[0]= [4.5] # Remember, it's still a matrix
    y_output_feature_tensor = torch.from_numpy(y_output_feature).unsqueeze(1)  # (N,) → (N, 1)    #Regression
    # Determine input and output sizes
    # Input_dim is the number of columns (number of features/number of independent variables)
    input_dim = x_input_features_tensor.shape[1]  # (n_samples, n_features) # (number_of_movies, number_of_genres)
    # Output_dim is the number of dependent variables (only 1 if it's a regression problem, multiple if it's a classification problem)
    output_dim = 1  # Regression
    # Return the tensors and dimensions
    # return x_input_features_tensor, y_output_feature_tensor, input_dim, output_dim
    # Return np.ndarray for skorch
    return x_input_features, y_output_feature, input_dim, output_dim


def _prepare_collaborative_filtering_training_data(
        dataset: pd.DataFrame, user_col: str = "userId", item_col: str = "movieId", target_col: str = "rating"
) -> Tuple[np.ndarray, np.ndarray, int, int, pd.Index, pd.Index]:
    """
    Prepares training data for collaborative filtering with embeddings.

    :param dataset:
    :param user_col:
    :param item_col:
    :param target_col:
    :return: Tuple[np.ndarray, np.ndarray, int, int, pd.Index, pd.Index]:
    """
    # Drop rows with missing ratings
    df = dataset.dropna(subset=[target_col])
    # Encode user and item IDs into contiguous integers
    user_ids, user_index = pd.factorize(df[user_col])  # userId → [0, 1, 2, ...]
    item_ids, item_index = pd.factorize(df[item_col])  # movieId → [0, 1, 2, ...]
    # Create input feature matrix: each row = [user_id, item_id]
    x_input_features = np.stack([user_ids, item_ids], axis=1).astype(np.int64)
    # Extract ratings and reshape to (N, 1)
    y_output_features = df[target_col].values.astype(np.float32)  # .reshape(-1, 1)
    # Determine number of unique users and items (for embedding layers)
    num_users = len(user_index)
    num_items = len(item_index)
    return x_input_features, y_output_features, num_users, num_items, user_index, item_index


class NeuralNetworkModelForContentBasedFiltering(nn.Module):
    """
    Define a neural network model by subclassing nn.Module (for skorch) CBF
    """
    # Constructor: receives input and output dimensions
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: List[int], dropout: float = 0.0):
        super().__init__()  # Initialize the base class (nn.Module)
        layers = []
        in_dim = input_dim
        # Dynamically create hidden layers
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))  # Add a fully connected layer
            layers.append(nn.ReLU())  # Add ReLU activation
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h  # Update input dimension for the next layer
        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))  # Output layer for regression or classification
        # Wrap all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # Forward pass through the network


class NeuralNetworkModelForCollaborativeFiltering(nn.Module):
    """
    Define a neural network model by subclassing nn.Module (for skorch) with embeddings for CF
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, hidden_dim: List[int], dropout: float = 0.0):
        super().__init__()
        # Embeddings for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        # Fully connected layers (MLP)
        layers = []
        input_dim = embedding_dim * 2  # concatenated user and item embeddings
        for h in hidden_dim:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # output layer: predicted rating
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: Long tensor of shape (batch_size, 2), columns = [user_id, item_id]
        user_ids = x[:, 0].long()
        item_ids = x[:, 1].long()
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        # Concatenate user and item embeddings
        x = torch.cat([user_embedded, item_embedded], dim=1)
        # Pass through MLP
        out = self.mlp(x)
        return out.squeeze(1)  # output shape: (batch_size,)


def _build_model(input_dim: Union[int, None], output_dim: Union[int, None], model_type: ModelType,
                 filtering_type: FilteringType) \
        -> Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]:
    """
    Builds the model

    :param input_dim:
    :param output_dim:
    :param model_type:
    :param filtering_type:
    :return: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]:
    """
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            if filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value or filtering_type.value == FilteringType.HYBRID_FILTERING.value :
                return _build_neural_network_model_for_content_based_filtering(input_dim, output_dim)
            else:
                return _build_neural_network_model_for_collaborative_filtering(input_dim, output_dim)
        case ModelType.RANDOM_FOREST.value:
            return _build_random_forest_model()
        case ModelType.XGBOOST.value:
            return _build_xgboost_model()
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def _build_neural_network_model_for_content_based_filtering(input_dims: int, output_dims: int,
                                                            saved_params: dict = None) -> NeuralNetRegressor:
    """
    Builds a nn for cbf

    :param input_dims:
    :param output_dims:
    :param saved_params:
    :return: NeuralNetRegressor:
    """
    model = NeuralNetRegressor(
        NeuralNetworkModelForContentBasedFiltering,  # Pytorch module class
        module__input_dim=input_dims,  # Number of independent variables
        module__output_dim=output_dims,  # Number of dependent variables (1)
        module__hidden_dim=saved_params[
            "module__hidden_dim"] if saved_params is not None else NeuralNetworkConfigForCBF.HIDDEN_DIM,
        module__dropout=saved_params[
            "module__dropout"] if saved_params is not None else NeuralNetworkConfigForCBF.DROPOUT,
        max_epochs=saved_params["max_epochs"] if saved_params is not None else NeuralNetworkConfigForCBF.NUM_EPOCHS,
        lr=saved_params["lr"] if saved_params is not None else NeuralNetworkConfigForCBF.LEARNING_RATE,
        batch_size=saved_params["batch_size"] if saved_params is not None else NeuralNetworkConfigForCBF.BATCH_SIZE,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=saved_params["iterator_train__shuffle"] if saved_params is not None else NeuralNetworkConfigForCBF.ITERATOR_TRAIN_SHUFFLE,
        device=saved_params["device"] if saved_params is not None else NeuralNetworkConfigForCBF.DEVICE,
        verbose=saved_params["verbose"] if saved_params is not None else NeuralNetworkConfigForCBF.VERBOSE
    )
    return model


def _build_neural_network_model_for_collaborative_filtering(num_users: int, num_items: int,
                                                            saved_params: dict = None) -> NeuralNetRegressor:
    """
    Builds a nn for cf

    :param num_users:
    :param num_items:
    :param saved_params:
    :return: NeuralNetRegressor:
    """
    model = NeuralNetRegressor(
        NeuralNetworkModelForCollaborativeFiltering,  # PyTorch module class
        module__num_users=num_users,  # Number of unique users
        module__num_items=num_items,  # Number of unique items
        module__embedding_dim=saved_params[
            "module__embedding_dim"] if saved_params is not None else NeuralNetworkConfigForCF.EMBEDDING_DIM,
        module__hidden_dim=saved_params[
            "module__hidden_dim"] if saved_params is not None else NeuralNetworkConfigForCF.HIDDEN_DIM,
        module__dropout=saved_params[
            "module__dropout"] if saved_params is not None else NeuralNetworkConfigForCF.DROPOUT,
        max_epochs=saved_params["max_epochs"] if saved_params is not None else NeuralNetworkConfigForCF.NUM_EPOCHS,
        lr=saved_params["lr"] if saved_params is not None else NeuralNetworkConfigForCF.LEARNING_RATE,
        batch_size=saved_params["batch_size"] if saved_params is not None else NeuralNetworkConfigForCF.BATCH_SIZE,
        optimizer= torch.optim.Adam,
        iterator_train__shuffle=saved_params["iterator_train__shuffle"] if saved_params is not None else NeuralNetworkConfigForCF.ITERATOR_TRAIN_SHUFFLE,
        device=saved_params["device"] if saved_params is not None else NeuralNetworkConfigForCF.DEVICE,
        verbose=saved_params["verbose"] if saved_params is not None else NeuralNetworkConfigForCF.VERBOSE,
    )
    return model


def _build_random_forest_model() -> RandomForestRegressor:
    """
    Builds a random forest model

    :return: RandomForestRegressor:
    """
    model = RandomForestRegressor(
        n_estimators=RandomForestConfig.N_ESTIMATORS,
        max_depth=RandomForestConfig.MAX_DEPTH,
        random_state=RandomForestConfig.RANDOM_STATE,
        verbose=RandomForestConfig.VERBOSE,
        n_jobs=RandomForestConfig.N_JOBS
    )
    return model


def _build_xgboost_model() -> xgb.XGBRegressor:
    """
    Builds a xgboost model

    :return: xgb.XGBRegressor:
    """
    model = xgb.XGBRegressor(
        n_estimators=XGBoostConfig.N_ESTIMATORS,
        max_depth=XGBoostConfig.MAX_DEPTH,
        random_state=XGBoostConfig.RANDOM_STATE,
        learning_rate=XGBoostConfig.LEARNING_RATE,
        tree_method=XGBoostConfig.TREE_METHOD,
        objective=XGBoostConfig.OBJECTIVE,
        verbosity=XGBoostConfig.VERBOSITY,
        n_jobs=XGBoostConfig.N_JOBS
    )
    return model


def _read_model_locally(
        model_type: ModelType,
        filtering_type: FilteringType,
        user_id: int,
        input_dim_or_num_users: Union[int, None],
        output_dim_or_num_items: Union[int, None]  # Only for Pytorch
) -> Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]:
    """
    # Reads model locally

    :param model_type:
    :param filtering_type:
    :param user_id:
    :param input_dim_or_num_users:
    :param output_dim_or_num_items:
    :return: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor]:
    """
    # Gets model path
    model_path: str = f"{FilePaths.get_model_path(model_type, filtering_type, user_id)}"
    # Read and return the model
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            with open(f"{model_path}_meta.json") as f:
                saved_params = json.load(f)
            # Build empty nn model
            if filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value or filtering_type.value == FilteringType.HYBRID_FILTERING.value:
                model: NeuralNetRegressor = _build_neural_network_model_for_content_based_filtering(
                    input_dims=input_dim_or_num_users,
                    output_dims=output_dim_or_num_items,
                    saved_params=saved_params)
            else:
                model: NeuralNetRegressor = _build_neural_network_model_for_collaborative_filtering(
                    num_users=input_dim_or_num_users,
                    num_items=output_dim_or_num_items,
                    saved_params=saved_params)
            model.initialize()
            # Load model with local data
            model.load_params(f_params=f"{model_path}.pkl")
            return model
        case ModelType.RANDOM_FOREST.value:
            model: RandomForestRegressor = joblib.load(f"{model_path}.pkl")
            return model
        case ModelType.XGBOOST.value:
            model: xgb.XGBRegressor = xgb.XGBRegressor()
            model_file = Path(f"{model_path}.json")
            if not model_file.exists():
                raise FileNotFoundError()
            model.load_model(f"{model_path}.json")
            return model
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def _save_model_locally(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        model_type: ModelType,
        filtering_type: FilteringType,
        user_id: int
):
    """
    Saves model locally

    :param model:
    :param model_type:
    :param filtering_type:
    :param user_id:
    :return:
    """
    # Gets model path
    model_path: str = f"{FilePaths.get_model_path(model_type, filtering_type, user_id)}"
    # Saves the model
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            # Save hyperparameters
            with open(f"{model_path}_meta.json", "w") as f:
                params: dict = filter_json_serializable(model.get_params())
                json.dump(params, f, indent=2)  # type: ignore
            # Save model
            model.save_params(f_params=f"{model_path}.pkl")
        case ModelType.RANDOM_FOREST.value:
            joblib.dump(model, f"{model_path}.pkl")
        case ModelType.XGBOOST.value:
            model.save_model(f"{model_path}.json")
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def _train_model(
        model: Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor],
        x_input_features: Union[torch.Tensor, np.ndarray],
        y_output_feature: Union[torch.Tensor, np.ndarray],
        model_type: ModelType,
        filtering_type: FilteringType,
        input_dim: Union[int, None],
        output_dim: Union[int, None],
) -> Union[Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], pd.DataFrame],
Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], None]]:
    """
    Trains the model

    :param model:
    :param x_input_features:
    :param y_output_feature:
    :param model_type:
    :param filtering_type:
    :param input_dim:
    :param output_dim:
    :return: Union[Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], pd.DataFrame],
Tuple[Union[NeuralNetRegressor, RandomForestRegressor, xgb.XGBRegressor], None]]:
    """
    match model_type.value:
        case ModelType.NEURAL_NETWORK.value:
            if GridSearchConfig.USE_GRID_SEARCH:
                if (filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value or
                        filtering_type.value == FilteringType.HYBRID_FILTERING.value):
                    best_model, gs_results_df = _grid_search_neural_network_for_cbf(model, x_input_features,
                                                                                    y_output_feature,
                                                                                    input_dim, output_dim)
                else:
                    best_model, gs_results_df = _grid_search_neural_network_for_cf(model, x_input_features,
                                                                                   y_output_feature,
                                                                                   input_dim, output_dim)
                return best_model, gs_results_df
            else:
                return _train_neural_network_model(model, x_input_features, y_output_feature), None
        case ModelType.RANDOM_FOREST.value:
            return _train_random_forest_model(model, x_input_features, y_output_feature), None
        case ModelType.XGBOOST.value:
            return _train_xgboost_model(model, x_input_features, y_output_feature), None
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")



def _train_neural_network_model(
        model: NeuralNetRegressor,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
) -> NeuralNetRegressor:
    """
    Trains the model using MSELoss and Adam optimizer. Returns trained model.

    :param model:
    :param x_input_features:
    :param y_output_feature:
    :return: NeuralNetRegressor:
    """
    model.fit(x_input_features.astype(np.float32), y_output_feature.astype(np.float32))
    return model


def _train_random_forest_model(
        model: RandomForestRegressor,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
) -> RandomForestRegressor:
    """
    Trains a random forest model

    :param model:
    :param x_input_features:
    :param y_output_feature:
    :return: RandomForestRegressor:
    """
    model.fit(x_input_features.astype(np.float32), y_output_feature.astype(np.float32))
    return model


def _train_xgboost_model(
        model: xgb.XGBRegressor,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
) -> xgb.XGBRegressor:
    """
    Trains a xgboost model

    :param model:
    :param x_input_features:
    :param y_output_feature:
    :return: xgb.XGBRegressor:
    """
    model.fit(x_input_features.astype(np.float32), y_output_feature.astype(np.float32))
    return model


def _grid_search_neural_network_for_cbf(
        empty_model: NeuralNetRegressor,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
        input_dim: int,
        output_dim: int
) -> Tuple[NeuralNetRegressor, pd.DataFrame]:
    """
    Uses grid search to find the best parameters for cbf

    :param empty_model:
    :param x_input_features:
    :param y_output_feature:
    :param input_dim:
    :param output_dim:
    :return: Tuple[NeuralNetRegressor, pd.DataFrame]:
    """
    param_grid = {
        'lr': NeuralNetworkGridSearchConfigForCBF.LEARNING_RATE,
        'max_epochs': NeuralNetworkGridSearchConfigForCBF.MAX_EPOCHS,
        'batch_size': NeuralNetworkGridSearchConfigForCBF.BATCH_SIZE,
        'module__hidden_dim': NeuralNetworkGridSearchConfigForCBF.HIDDEN_DIM,
        'module__dropout': NeuralNetworkGridSearchConfigForCBF.DROPOUT,
        'module__input_dim': [input_dim],
        'module__output_dim': [output_dim]
    }
    gs = GridSearchCV(
        empty_model,
        param_grid,
        cv=NeuralNetworkGridSearchConfigForCBF.CV,
        scoring=NeuralNetworkGridSearchConfigForCBF.SCORING,
        verbose=NeuralNetworkGridSearchConfigForCBF.VERBOSE,
        return_train_score=True,
        n_jobs=NeuralNetworkGridSearchConfigForCBF.N_JOBS
    )
    gs.fit(x_input_features.astype(np.float32), y_output_feature.astype(np.float32))
    write_log(f"Grid search complete. Best params: {gs.best_params_}")
    return gs.best_estimator_, pd.DataFrame(gs.cv_results_)


def _grid_search_neural_network_for_cf(
        empty_model: NeuralNetRegressor,
        x_input_features: np.ndarray,
        y_output_feature: np.ndarray,
        num_users: int,
        num_items: int
) -> Tuple[NeuralNetRegressor, pd.DataFrame]:
    """
    Uses grid search to find the best parameters for collaborative filtering

    :param empty_model:
    :param x_input_features:
    :param y_output_feature:
    :param num_users:
    :param num_items:
    :return: Tuple[NeuralNetRegressor, pd.DataFrame]:
    """
    param_grid = {
        'lr': NeuralNetworkGridSearchConfigForCF.LEARNING_RATE,
        'max_epochs': NeuralNetworkGridSearchConfigForCF.MAX_EPOCHS,
        'batch_size': NeuralNetworkGridSearchConfigForCF.BATCH_SIZE,
        'module__embedding_dim': NeuralNetworkGridSearchConfigForCF.EMBEDDING_DIM,
        'module__hidden_dim': NeuralNetworkGridSearchConfigForCF.HIDDEN_DIM,
        'module__dropout': NeuralNetworkGridSearchConfigForCF.DROPOUT,
        'module__num_users': [num_users],
        'module__num_items': [num_items]
    }

    gs = GridSearchCV(
        empty_model,
        param_grid,
        cv=NeuralNetworkGridSearchConfigForCF.CV,
        scoring=NeuralNetworkGridSearchConfigForCF.SCORING,
        verbose=NeuralNetworkGridSearchConfigForCF.VERBOSE,
        return_train_score=True,
        n_jobs=NeuralNetworkGridSearchConfigForCF.N_JOBS
    )

    gs.fit(x_input_features.astype(np.int64), y_output_feature.astype(np.float32))
    write_log(f"Grid search complete. Best params: {gs.best_params_}")
    return gs.best_estimator_, pd.DataFrame(gs.cv_results_)


def _save_gs_results_locally(
        gs_results: pd.DataFrame,
        model_type: ModelType,
        filtering_type: FilteringType,
        user_id: int
):
    """
    Saves grid search results locally

    :param gs_results:
    :param model_type:
    :param filtering_type:
    :param user_id:
    :return:
    """
    # Gets model path
    gs_path: str = f"{FilePaths.get_model_path(model_type, filtering_type, user_id)}_grid_search_results.csv"
    # Saves the grid search results
    gs_results.to_csv(gs_path, index=False)
