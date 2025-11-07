import math
import random

import pandas as pd
import streamlit as st

from configs import FilteringType
from core import data_loader, external_api, add_data
from core.dtos.added_rating_dto import AddedRatingDTO
from core.dtos.added_user_dto import AddedUserDTO
from ui import ui_helpers, app_data_loader, app_prediction, app_recommendation, app_explainability
from ui.dtos.ui_movies_cbf_global_explanation_container_dto import UiMoviesCbfGlobalExplanationContainerDto
from ui.dtos.ui_user_page_dto import UiUserPageDto

if __name__ == '__main__':
    pass


def user_page(
        ui_user_page_dto: UiUserPageDto,
        selected_filtering_type: FilteringType,  # only needed for explanation
        ui_movies_cbf_global_explanation_container_dto: UiMoviesCbfGlobalExplanationContainerDto = None,
):
    """
    Page with the selected user profile

    :param ui_user_page_dto:
    :param selected_filtering_type:
    :param ui_movies_cbf_global_explanation_container_dto:
    :return:
    """
    # Gets the CBF rated dataframe (with users 1-5, or top 5)
    cbf_rated_df_for_user_selection, _ = app_data_loader.initialize_dataframes(
        ratings_user_id=-1,
        clean_and_index_dfs=False,
        filtering_type=FilteringType.CONTENT_BASED_FILTERING
    )
    # Get list of unique users (1-5, or top 5)
    user_ids = app_data_loader.get_user_ids_from_rated_df(cbf_rated_df_for_user_selection)

    # If selected filtering type is hybrid, allow to select from added_user as well
    if selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value:
        added_users_df = data_loader.get_added_users_dataframe()["userId"].to_frame()
        user_ids = pd.concat(
            [user_ids,
             added_users_df
             ], ignore_index=True
        )

    col1, col2 = st.columns([1, 2])

    with (col1):
        st.subheader("👤 User")
        st.text(f"Selected user id: {int(float(ui_user_page_dto.selected_user))}")
        # Sidebar: user selector
        selected_user = ui_helpers.user_selector_helper(user_ids, display_on_sidebar=False)

        st.markdown("---")

        st.subheader("🎬 Actions")

        # Add/Create new user function
        def on_add_user_callback():
            # Get an available userId  # Random id between 100M and 200M (low risk of collision)
            next_unique_available_user_id: int = random.randint(100000000, 200000000)
            # Call function to write user to csv
            add_data.add_user_to_csv(added_user_dto=AddedUserDTO(userId=next_unique_available_user_id))
            # Activate success message
            st.session_state["added_user"] = next_unique_available_user_id
            # Add random interaction (avoid cold start problem)
            add_data.add_rating_to_csv(
                added_rating_dto=AddedRatingDTO(
                    userId=next_unique_available_user_id,
                    movieId=1,
                    rating=2.5)
            )
            # Maybe change selected user to added user?
        disabled_add_user_button:bool = not selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value
        st.button(
            label="➕ Create new User",
            on_click=on_add_user_callback,
            disabled=disabled_add_user_button or ("show_global_explanation" in st.session_state)
        )
        if disabled_add_user_button:
            st.warning("Only Hybrid Filtering supports creating new users")

        # If added user, show success message
        if "added_user" in st.session_state:
            st.success(f"Successfully added user with userId:{st.session_state["added_user"]} ✅")
            st.session_state.pop("added_user", None)

        if not "show_global_explanation" in st.session_state:
            st.button(
                label="📊 Show global recommendation explanation",
                on_click=lambda: st.session_state.update(
                    {"show_global_explanation": True}),
                disabled=True if selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value else False,
            )
            if selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
                st.warning(
                    f"{selected_filtering_type.value.replace("_", " ").capitalize()} doesn't support global explanations.")

        else:
            st.button(
                label="📊 Hide global recommendation explanation",
                on_click=lambda: st.session_state.pop("show_global_explanation", None),
            )

    with col2:
        st.subheader("🌟 Ratings")

        st.text(
            f"User {int(float(ui_user_page_dto.selected_user))} has rated {ui_user_page_dto.number_of_unique_rated_movies_by_the_user} unique movies.")

        movies_rated_by_selected_user_df = ui_user_page_dto.movies_rated_by_selected_user_df.copy()
        # Turn timestamp into date
        # movies_rated_by_selected_user_df['date'] = pd.to_datetime(
        #    movies_rated_by_selected_user_df['timestamp'], unit='s')
        # Format date
        # movies_rated_by_selected_user_df['formated_date'] = movies_rated_by_selected_user_df['date'].dt.strftime(
        #    '%Y-%m-%d')

        # Order by date
        movies_rated_by_selected_user_df = movies_rated_by_selected_user_df.sort_values(by='timestamp', ascending=False)

        # Grab first few lines and only the important columns
        movies_rated_by_selected_user_display_df = movies_rated_by_selected_user_df[
                                                       ['title', 'rating', 'timestamp']][
                                                   0:30]  # 'date', 'formated_date'

        st.write(movies_rated_by_selected_user_display_df)
    st.markdown("---")

    if "show_global_explanation" in st.session_state:
        movies_global_explanation_container(
            selected_filtering_type=selected_filtering_type,
            ui_movies_cbf_global_explanation_container_dto=ui_movies_cbf_global_explanation_container_dto,
        )


def movies_global_explanation_container(
        selected_filtering_type: FilteringType,
        ui_movies_cbf_global_explanation_container_dto: UiMoviesCbfGlobalExplanationContainerDto = None,
):
    """
    Container with the global explanations about the recommendations of the selected user profile

    :param selected_filtering_type:
    :param ui_movies_cbf_global_explanation_container_dto:
    :return:
    """
    # Show Local SHAP explanation
    match selected_filtering_type.value:
        # Show Local explanation for CBF
        case FilteringType.CONTENT_BASED_FILTERING.value | FilteringType.HYBRID_FILTERING.value:
            # Clean selected movie for SHAP explanation
            selected_movie = (
                ui_movies_cbf_global_explanation_container_dto.predicted_movies_to_explain_df.copy()).reset_index()
            _, selected_movie = data_loader._clean_and_index_dfs(selected_movie, selected_movie,
                                                                 FilteringType.CONTENT_BASED_FILTERING)

            # Generate SHAP explanation for a single movie (local explainability)
            filtered_shap_values_local_explanation, filtered_feature_names_local_explanation = app_explainability.generate_shap_explanation_and_filter_out_zero_importance_features(
                prediction_model=ui_movies_cbf_global_explanation_container_dto.prediction_model,
                rated_movies_dataset=ui_movies_cbf_global_explanation_container_dto.rated_df,
                movies_to_explain_df=selected_movie,
                global_or_local="global",
                model_type=ui_movies_cbf_global_explanation_container_dto.selected_model_type
            )
            st.markdown("#### 💡 SHAP Explanation (Global)")

            # Charts side by side
            shap_local_chart_col1, shap_local_chart_col2 = st.columns(2)
            fig_bar_local, fig_pie_local = app_explainability.get_shap_bar_and_pie_chart_figures(
                filtered_shap_values_local_explanation, filtered_feature_names_local_explanation, "global")
            # Render charts
            col_a1, col_a2, _ = st.columns(3)

            ui_helpers.render_shap_explanation_charts(fig_bar_local, fig_pie_local)
            st.markdown("---")

        case _:
            raise ValueError(
                f"Unsupported filtering type: {selected_filtering_type}, type={type(selected_filtering_type)}")
