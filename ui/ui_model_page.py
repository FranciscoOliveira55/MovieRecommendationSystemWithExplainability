import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from configs import FilteringType, ModelType, FilePaths, LEARNINGCURVEConfig
from core import data_loader, external_api
from ui import ui_helpers, app_data_loader, app_prediction, app_recommendation, app_explainability, app_evaluation
from ui.dtos.ui_model_analytics_and_comparison_container_dto import UiModelAnalyticsAndComparisonContainerDto
from ui.dtos.ui_model_page_dto import UiModelPageDto

if __name__ == '__main__':
    pass


def model_page(
        ui_model_page_dto: UiModelPageDto,
        ui_model_analytics_and_comparison_container_dto: UiModelAnalyticsAndComparisonContainerDto,
):
    """
    Page of the models

    :param ui_model_page_dto:
    :param ui_model_analytics_and_comparison_container_dto:
    :return:
    """


    col1, col2 = st.columns([1, 2])

    with col1:
        model_path:Path = Path(FilePaths.get_model_path(
            model_type=ui_model_page_dto.selected_model_type,
            filtering_type=ui_model_page_dto.selected_filtering_type,
            selected_user_id=int(ui_model_page_dto.selected_user)
        ))
        st.subheader("💡 Model")
        st.caption(f"Using model: {model_path.parents[1].name} / {model_path.parents[0].name} / {model_path.name}")
        #st.markdown("---")

        #st.text(f"Selected filtering type: {ui_model_page_dto.selected_filtering_type}")
        # Selected filtering type
        selected_filtering_type: FilteringType = ui_helpers.filtering_type_selector_helper(display_on_sidebar=False)
        #st.markdown("---")

        #st.text(f"Selected model type: {ui_model_page_dto.selected_model_type}")
        # Selected model type
        selected_model_type: ModelType = ui_helpers.model_type_selector_helper(display_on_sidebar=False)
        st.markdown("---")

        st.subheader("🎬 Actions")
        if not "show_model_analytics_and_comparison" in st.session_state:
            st.button(
                label="📊 Show model analytics and comparison",
                on_click=lambda: st.session_state.update(
                    {"show_model_analytics_and_comparison": True})
            )
        else:
            st.button(
                label="📊 Hide Show model analytics and comparison",
                on_click=lambda: st.session_state.pop("show_model_analytics_and_comparison", None)
            )

    with col2:
        filtering_type_explanation = {
            f'{FilteringType.CONTENT_BASED_FILTERING.value}': 'Content-Based Filtering (CBF): Recommends items similar to those the user has already rated positively, based on their attributes. Advantage: highly personalized to specific tastes. Limitation: may fail to suggest very different or novel items.',
            f'{FilteringType.COLLABORATIVE_FILTERING.value}': 'Collaborative Filtering (CF): Recommends items based on the preferences of similar users. Advantage: uncovers patterns beyond item attributes. Limitation: suffers from the “cold start” problem for new users or items.',
            f'{FilteringType.HYBRID_FILTERING.value}': 'Hybrid Filtering: Combines CBF and CF to leverage the strengths of both. Advantage: provides more balanced recommendations and mitigates individual limitations. Limitation: may require more computational resources and complexity.',
            f'{FilteringType.CONTENT_BASED_FILTERING.value}_image': 'https://cdn.sanity.io/images/oaglaatp/production/a2fc251dcb1ad9ce9b8a82b182c6186d5caba036-1200x800.png?w=1200&h=800&auto=format',
            f'{FilteringType.COLLABORATIVE_FILTERING.value}_image': 'https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/04/user-based-collaborative-filtering.jpg?fit=1200%2C675&ssl=1',
            f'{FilteringType.HYBRID_FILTERING.value}_image': 'https://www.muvi.com/wp-content/uploads/2022/04/Hybrid-Filtering.jpg'
        }

        model_type_explanation = {
            f'{ModelType.NEURAL_NETWORK.value}': 'Neural Network: Learns complex, non-linear relationships in the data through layers of interconnected nodes. Advantage: highly flexible and powerful for capturing intricate patterns. Limitation: requires more data and computational power.',
            f'{ModelType.RANDOM_FOREST.value}': 'Random Forest: An ensemble of decision trees that votes on predictions. Advantage: robust to overfitting and works well with varied data. Limitation: can be less interpretable and slower for large datasets.',
            f'{ModelType.XGBOOST.value}': 'XGBoost: Extreme gradient boosting algorithm optimized for speed and performance. Advantage: often achieves high accuracy with structured data. Limitation: may require careful tuning to avoid overfitting.',
            f'{ModelType.NEURAL_NETWORK.value}_image': 'https://images.ctfassets.net/7p3vnbbznfiw/5jCkun4Xm2AohPMW5IEcyD/a5fc6245a03768413855a05fc22f3a04/neural-network-process.png',
            f'{ModelType.RANDOM_FOREST.value}_image': 'https://images.prismic.io/turing/65980c06531ac2845a27269b_Random_Forest_Algorithm_400798756c.webp?auto=format,compress',
            f'{ModelType.XGBOOST.value}_image': 'https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/xgboost/img-3.png'
        }

        st.subheader("ℹ️ Description")
        #st.markdown("---")

        col3, col4 = st.columns(2)
        #Show selected filtering type description
        with col3:
            st.image(filtering_type_explanation[f'{selected_filtering_type.value}_image'], use_container_width=True)
            st.text(filtering_type_explanation[selected_filtering_type.value])
        with col4:
            st.image(model_type_explanation[f'{selected_model_type.value}_image'], use_container_width=True)
            st.text(model_type_explanation[selected_model_type.value])
    st.markdown("---")

    # If show_model_analytics_and_comparison in session (button clicked) show model cross validation and learning curve
    if "show_model_analytics_and_comparison" in st.session_state:
        model_analytics_and_comparison_container(
            ui_model_analytics_and_comparison_container_dto=ui_model_analytics_and_comparison_container_dto,
        )


def model_analytics_and_comparison_container(
        ui_model_analytics_and_comparison_container_dto: UiModelAnalyticsAndComparisonContainerDto,
):
    """
    Container of the model analytics and comparison (with the learning curve and cross validation results)

    :param ui_model_analytics_and_comparison_container_dto:
    :return:
    """
    # Calculate learning curve for selected model
    if "learning_curve_scoring_metric_key" in st.session_state:
        learning_curve_scoring_metric_key = st.session_state["learning_curve_scoring_metric_key"]
    else:
        learning_curve_scoring_metric_key = list(LEARNINGCURVEConfig.SCORING_METRICS.keys())[0]
        st.session_state["learning_curve_scoring_metric_key"] = learning_curve_scoring_metric_key

    learning_curve_dict: dict = app_evaluation.calc_learning_curve_of_model_for_evaluation(
        prediction_model=ui_model_analytics_and_comparison_container_dto.prediction_model,
        rated_df=ui_model_analytics_and_comparison_container_dto.rated_df,
        selected_user=ui_model_analytics_and_comparison_container_dto.selected_user,
        selected_model=ui_model_analytics_and_comparison_container_dto.selected_model_type,
        filtering_type=ui_model_analytics_and_comparison_container_dto.selected_filtering_type,
        learning_curve_scoring_metric_key=learning_curve_scoring_metric_key
    )
    # Make cross validation for the selected model
    cv_results_dict: dict = app_evaluation.cross_validate_model_for_evaluation(
        prediction_model=ui_model_analytics_and_comparison_container_dto.prediction_model,
        rated_df=ui_model_analytics_and_comparison_container_dto.rated_df,
        selected_user=ui_model_analytics_and_comparison_container_dto.selected_user,
        selected_model=ui_model_analytics_and_comparison_container_dto.selected_model_type,
        filtering_type=ui_model_analytics_and_comparison_container_dto.selected_filtering_type
    )
    # Prepare learning curve for display
    learning_curve_fig: Figure = app_evaluation.prepare_learning_curve_for_display(learning_curve_dict)
    # Prepare cv results for display
    cv_metrics_per_fold_df, summary_mean_and_standard_deviation_df, cv_performance_per_fold_chart_figure, cv_variance_per_metric_figure = app_evaluation.prepare_cross_validation_results_for_display(
        cv_results_dict)
    # Display learning curve and cv results
    ui_helpers.render_learning_curve_and_cross_validation_results(
        learning_curve_fig=learning_curve_fig,
        cv_metrics_per_fold_df=cv_metrics_per_fold_df,
        summary_mean_and_standard_deviation_df=summary_mean_and_standard_deviation_df,
        cv_performance_per_fold_chart_figure=cv_performance_per_fold_chart_figure,
        cv_variance_per_metric_figure=cv_variance_per_metric_figure
    )