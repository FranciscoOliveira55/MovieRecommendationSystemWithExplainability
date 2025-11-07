import pandas as pd
import streamlit as st
from attr.validators import disabled
from matplotlib.figure import Figure
from torch.compiler import disable

from configs import ModelType, LEARNINGCURVEConfig, FilteringType
from core import data_loader

if __name__ == '__main__':
    pass


def user_selector_helper(user_ids: pd.DataFrame, display_on_sidebar=True) -> str:
    """
    Sidebar selector for user ID and session state management.

    :param user_ids:
    :param display_on_sidebar:
    :return: str:
    """
    # Prepare user options
    user_options = list(user_ids.itertuples(index=False))
    # Grab current selected user option (if selected_user in session)
    previously_selected_user = st.session_state.get("selected_user", None)
    default_value_index = 0
    if previously_selected_user is not None:
        for i, item in enumerate(user_options):
            if item.userId == previously_selected_user:
                default_value_index = i
                break  # Make select

    def on_change_callback():
        new_selected_user = st.session_state["_temp_selected_user"].userId
        st.session_state["selected_user"] = new_selected_user
        for key in [
            "generate_recommendations_clicked",
            "rated_df",
            "prediction_model",
            "predicted_rating_df",
            "selected_movie_index",
            "background_data",
            "shap_values_global_explanation",
            "shap_values_local_explanation"]:
            st.session_state.pop(key, None)
        # write_log(f"Saved selected_user in session {new_selected_user}")

    # Make select box
    select_box = st.sidebar.selectbox if display_on_sidebar else st.selectbox
    selected_user = (select_box(
        label="",  # "Select different user",
        options=user_options,
        format_func=lambda x: f"User id: {int(x.userId)}" + (
            f" number of rated movies: {int(x.numberOfRatedMovies)}" if not pd.isna(
                x.numberOfRatedMovies) else "(added user)"),
        index=default_value_index,
        key="_temp_selected_user",
        on_change=on_change_callback,
        disabled=("show_global_explanation" in st.session_state)
    )).userId  # save just userId

    return selected_user


def filtering_type_selector_helper(display_on_sidebar=True) -> FilteringType:
    """
    Sidebar selector for Content based filtering or collaborative filtering and session state management.

    :param display_on_sidebar:
    :return: FilteringType:
    """
    # Prepare filtering type options
    filtering_type_options: list[FilteringType] = list(FilteringType)
    # Grab current selected filtering type option (if selected_filtering_type in session)
    previously_selected_filtering_type: FilteringType = st.session_state.get("selected_filtering_type", None)
    default_value_index = 0  # Content based filtering
    if previously_selected_filtering_type is not None:
        for i, filtering_type in enumerate(filtering_type_options):
            if filtering_type.value == previously_selected_filtering_type.value:
                default_value_index = i
                break  # Make select

    def on_change_callback():
        new_selected_filtering_type: FilteringType = st.session_state["_temp_selected_filtering_type"]
        st.session_state["selected_filtering_type"] = new_selected_filtering_type
        # Force model type to neural network if filtering type is collaborative filtering
        if new_selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
            st.session_state["selected_model_type"] = ModelType.NEURAL_NETWORK
        for key in [
            "generate_recommendations_clicked",
            "rated_df",
            "prediction_model",
            "predicted_rating_df",
            "selected_movie_index",
            "background_data",
            "shap_values_global_explanation",
            "shap_values_local_explanation",
            "show_model_analytics_and_comparison"
        ]:
            st.session_state.pop(key, None)

    # Disable filtering selection if an added user is selected (as well as hybrid filtering)
    disabled_filtering_selection_box = False
    if ("selected_filtering_type" in st.session_state) and ("selected_user" in st.session_state):
        selected_filtering_type: FilteringType = st.session_state["selected_filtering_type"]
        selected_user: int = int(st.session_state["selected_user"])
        if selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value:
            added_users_list: list[int] = data_loader.get_added_users_dataframe()["userId"].astype(int).tolist()
            if selected_user in added_users_list:
                disabled_filtering_selection_box = True

    # Make select box
    select_box = st.sidebar.selectbox if display_on_sidebar else st.selectbox
    selected_filtering_type = (select_box(
        label="Filtering Type",  # "Select different filtering type",
        options=filtering_type_options,
        format_func=lambda x: x.value.replace("_", " ").title(),
        index=default_value_index,
        key="_temp_selected_filtering_type",
        on_change=on_change_callback,
        disabled=disabled_filtering_selection_box or ("show_model_analytics_and_comparison" in st.session_state)
    ))
    if disabled_filtering_selection_box:
        st.warning("'Added users' are only supported by hybrid filtering.")

    return selected_filtering_type


def model_type_selector_helper(display_on_sidebar=True) -> ModelType:
    """
    Sidebar selector for ai recommendation model

    :param display_on_sidebar:
    :return: ModelType:
    """
    # Prepare model type options
    model_type_options: list[ModelType] = list(ModelType)
    # Grab current selected model type option (if selected_model_type in session)
    previously_selected_model_type: ModelType = st.session_state.get("selected_model_type", None)
    default_value_index = 0  # Neural network
    if previously_selected_model_type is not None:
        for i, model_type in enumerate(model_type_options):
            if model_type.value == previously_selected_model_type.value:
                default_value_index = i
                break  # Make select
    currently_selected_filtering_type: FilteringType = st.session_state.get("selected_filtering_type", None)

    def on_change_callback():
        new_selected_model_type = st.session_state["_temp_selected_model_type"]
        st.session_state["selected_model_type"] = new_selected_model_type
        for key in [
            "generate_recommendations_clicked",
            "rated_df",
            "prediction_model",
            "predicted_rating_df",
            "selected_movie_index",
            "background_data",
            "shap_values_global_explanation",
            "shap_values_local_explanation",
            "show_model_analytics_and_comparison"]:
            st.session_state.pop(key, None)

    # Make select box
    select_box = st.sidebar.selectbox if display_on_sidebar else st.selectbox
    selected_model_type = (select_box(
        label="Model Type",  # "Select different filtering type",
        options=model_type_options,
        format_func=lambda x: x.value.replace("_", " ").title(),
        index=default_value_index,
        key="_temp_selected_model_type",
        on_change=on_change_callback,
        disabled=currently_selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value or ("show_model_analytics_and_comparison" in st.session_state)
    ))

    return selected_model_type


def generate_recommendations_button() -> ():
    """
    Generate recommendations button

    :return:
    """
    if st.sidebar.button("🎯 Generate Recommendations"):
        st.session_state["generate_recommendations_clicked"] = True
        keys_to_clear = [
            "model_analytics_and_comparison_clicked",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)


def generate_model_analytics_and_comparison_button() -> ():
    """
    Generate model analytics and comparison button

    :return:
    """
    if st.sidebar.button("📊 Model Analytics and Comparison"):
        st.session_state["model_analytics_and_comparison_clicked"] = True
        keys_to_clear = [
            "generate_recommendations_clicked",
            "cv_results"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)


def render_shap_explanation_charts(fig_bar, fig_pie):
    """
    Renders the shap explanation charts

    :param fig_bar:
    :param fig_pie:
    :return:
    """
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        _, col_b1, _ = st.columns([1, 3, 1])
        with col_b1:
            st.pyplot(fig_bar, use_container_width=False)
        st.markdown(
            """
            **ℹ️ Bar Chart Explanation:**

            - Each bar represents a feature (e.g., movie genre, presence of an actor, etc.).
            - The height of the bar shows how much that feature contributed to the model’s prediction.
            - Positive values indicate the feature increased the predicted rating, while negative values indicate it decreased the predicted rating.
            - Even if a movie does not have a certain feature (for example, it’s not a specific genre), the bar can still have a non-zero value. This happens because the model considers the impact of the absence of the feature on the prediction compared to the average.
            """
        )
    with col_a2:
        _, col_c1, _ = st.columns([1, 3, 1])
        with col_c1:
            st.pyplot(fig_pie, use_container_width=False)
        st.markdown(
            """
            **ℹ️ Pie Chart Explanation:**

            - Each slice shows the relative contribution of each feature to the model’s prediction.
            - The size of the slice indicates how much that feature influenced the prediction relative to the total.
            - Even features that the movie does not have can appear as slices because the model accounts for the effect of both presence and absence.
            - Very small slices can be grouped into “Other” to simplify visualization.
            - Unlike the bar chart, the pie chart shows magnitude only, not effect direction.
            """
        )


def render_learning_curve_and_cross_validation_results(
        learning_curve_fig: Figure,
        cv_metrics_per_fold_df: pd.DataFrame,
        summary_mean_and_standard_deviation_df: pd.DataFrame,
        cv_performance_per_fold_chart_figure: Figure,
        cv_variance_per_metric_figure: Figure
):
    """
    Renders the learning curve and cross validation results

    :param learning_curve_fig:
    :param cv_metrics_per_fold_df:
    :param summary_mean_and_standard_deviation_df:
    :param cv_performance_per_fold_chart_figure:
    :param cv_variance_per_metric_figure:
    :return:
    """
    st.subheader("💡 Performance Evaluation")
    ##st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📉 Learning Curve")
        _, col_a1, _ = st.columns([1, 3, 1])
        with col_a1:
            st.pyplot(learning_curve_fig)

        st.markdown(
            """            
            The **learning curve** illustrates how the performance of the model changes as the amount of training data increases.
            It helps detecting problems like underfitting or overfitting.
            - **Underfitting** occurs when both training and validation performance remain low;
            - **Overfitting** happens when training performance is high but validation performance stays low.
            """
        )

    with col2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🖍️ Chosen metric:")
            # Pick learning curve scoring metric key
            st.selectbox(
                "Choose the learning curve scoring metric",
                options=list(LEARNINGCURVEConfig.SCORING_METRICS.keys()),
                key="learning_curve_scoring_metric_key",
            )
        scoring_metric_explanation = {
            'mse': '- **Mean Squared Error (MSE)**: Penalizes large errors by squaring them. Lower is better.',
            'rmse': '- **Root Mean Squared Error (RMSE)**: Square root of MSE. Easier to interpret as it is in the same units as the target.',
            'mae': '- **Mean Absolute Error (MAE)**: Measures average absolute differences between predicted and actual values. More robust to outliers.',
            'r2': '- **R² Score (Coefficient of Determination)**: Indicates how well the model explains the variance in the target. 1 is perfect, 0 means no explanatory power, negative value means worse predicting power than just the average of the target values.'
        }
        st.markdown("---")
        st.markdown("**ℹ️ Chosen metric description:**")
        for metric in list(LEARNINGCURVEConfig.SCORING_METRICS.keys()):
            st.caption(f"{scoring_metric_explanation[metric]}")

    st.markdown("---")
    col_b1, col_b2 = st.columns([4, 3])
    with col_b1:
        st.markdown("#### 📊 Cross-Validation Metrics Per Fold:")
        # ----- 1. Table per fold -----
        st.dataframe(cv_metrics_per_fold_df.style.format(precision=4), use_container_width=True)
        # ----- 2. Summary Mean ± standard deviation -----
        st.dataframe(summary_mean_and_standard_deviation_df, use_container_width=True)
    with col_b2:
        st.markdown(
            """
            #### ℹ️ Explanation:  
            
            **Cross-validation** is a technique used to evaluate how well a model **generalizes** to unseen data.  The data is split into several fold, the model is trained on some folds and tested on the remaining one, ensuring that every sample is used for both training and validation.  
            This helps detect **overfitting** and provides a more reliable estimate of model performance.  
            """
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        # ----- 3. Comparative chart (Bar Plot or Boxplot) -----
        _, col_c1, _ = st.columns([1, 3, 1])
        with col_c1:
            st.pyplot(cv_performance_per_fold_chart_figure)
    with col2:
        # Boxplot for variability
        _, col_d1, _ = st.columns([1, 3, 1])
        with col_d1:
            st.pyplot(cv_variance_per_metric_figure)
    st.markdown("---")
