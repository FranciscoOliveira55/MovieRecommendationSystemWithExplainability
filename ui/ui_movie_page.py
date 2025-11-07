import math
from datetime import datetime
from typing import Union

import pandas as pd
import streamlit as st
from narwhals.stable.v1 import Datetime
from sklearn.ensemble import RandomForestRegressor
from skorch import NeuralNetRegressor
import xgboost as xgb

from configs import FilteringType, ModelType
from core import data_loader, external_api, explainability, add_data
from core.dtos.added_rating_dto import AddedRatingDTO
from core.utils import write_log
from ui import ui_helpers, app_data_loader, app_prediction, app_recommendation, app_explainability
from ui.dtos.ui_movie_cbf_local_explanation_container_dto import UiMovieCbfLocalExplanationContainerDto
from ui.dtos.ui_movie_cf_local_explanation_container_dto import UiMovieCfLocalExplanationContainerDto
from ui.dtos.ui_movie_hyb_local_explanation_container_dto import UiMovieHybLocalExplanationContainerDto
from ui.dtos.ui_movie_page_dto import UiMoviePageDto

if __name__ == '__main__':
    pass


def movie_page(
        ui_movie_page_dto: UiMoviePageDto,
        selected_filtering_type: FilteringType,  # only needed for explanation
        ui_movie_cbf_local_explanation_container_dto: UiMovieCbfLocalExplanationContainerDto = None,
        ui_movie_cf_local_explanation_container_dto: UiMovieCfLocalExplanationContainerDto = None,
        ui_movie_hyb_local_explanation_container_dto: UiMovieHybLocalExplanationContainerDto = None
):
    """
    Page with the selected movie information and interactions

    :param ui_movie_page_dto:
    :param selected_filtering_type:
    :param ui_movie_cbf_local_explanation_container_dto:
    :param ui_movie_cf_local_explanation_container_dto:
    :param ui_movie_hyb_local_explanation_container_dto:
    :return:
    """
    selected_movie: pd.DataFrame = ui_movie_page_dto.selected_movie_df
    selected_mode: str = ui_movie_page_dto.selected_mode

    if selected_movie.empty:
        st.warning("Movie details not available.")
        return

    # Desired column order (put predictedRating second)
    #desired_order = ['title', 'PredictedRating'] + [col for col in selected_movie.columns if
    #                                                col not in ['title', 'PredictedRating']]
    #selected_movie = selected_movie[desired_order]

    # Extrair dados
    movie = selected_movie.iloc[0]
    title = movie.get('title', 'Unknown title')
    rating = movie.get('rating', 'N/A')
    predicted_rating = movie.get('PredictedRating', 'N/A')
    release_date = movie.get('release_date', 'No release_date')
    overview = movie.get('overview', 'No overview')
    poster_url = movie.get('poster_url', None)
    popularity = movie.get('popularity', 'N/A')
    vote_average = movie.get('vote_average', 'N/A')
    movie_genres = [col for col in movie.index if movie[col] == 1 and (
            col not in ["userId", "movieId"])]  # It is a movie's genre if its cell value is 1
    # Gerar HTML para cada género como badge circular
    genre_badges = "".join([
        f"""
        <span style="
            display:inline-block;
            background-color:#444;
            color:white;
            padding:6px 12px;
            margin:3px;
            border-radius:20px;
            font-size:0.9rem;
        ">{g}</span>
        """
        for g in movie_genres
    ])

    # Layout: 2 colunas
    col_a1, col_a2 = st.columns([1, 3])

    # Coluna 1: Poster
    with col_a1:
        if "selected_movie_id" in st.session_state:
            st.button(
                label="🔙 Back",
                on_click=lambda: [st.session_state.pop(k, None) for k in
                                  ("selected_movie_id", "show_local_explanation", "selected_page", "add_rating", "selected_mode", "selected_genres", "search_query")]
            )

        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.image("https://www.envirochoice.com.au/Images/ProductImages/product-image-1.png", use_container_width=True)

    # Coluna 2: Detalhes
    with col_a2:
        st.markdown(f"### {title}")
        st.caption(f"Release date: {release_date}")
        genres_col, _ = st.columns([3, 1])
        with genres_col:
            genre_cols = st.columns(len(movie_genres))
            for col, genre in zip(genre_cols, movie_genres):
                with col:
                    st.markdown(
                        f"<div style='background-color:#444;color:white;padding:6px;border-radius:20px;text-align:center;'>{genre}</div>",
                        unsafe_allow_html=True)
        st.markdown("---")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            match selected_mode:
                case "recommended_movies":
                    st.metric("Predicted Rating", f"{predicted_rating:.1f} ⭐")
                case "rated_movies":
                    st.metric("Given Rating", f"{rating:.1f} ⭐")
                case _:
                    st.error(f"Mode of movie to display invalid: {selected_mode}, type={type(selected_mode)}")

        with col_b2:
            st.metric("TMDB Rating", f"{vote_average} 🎬")
        with col_b3:
            st.metric("Popularity", f"{popularity} 🍿")

        st.markdown("#### Overview")
        st.write(overview)

        st.subheader("🎬 Actions")
        if not "add_rating" in st.session_state:
            col_c1, col_c2, _ = st.columns(3)
            with col_c1:
                st.button(
                    label="🌟 Add rating",
                    on_click=lambda: st.session_state.update(
                        {"add_rating": True}),
                    disabled="show_local_explanation" in st.session_state or (selected_mode == "rated_movies") or (selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value)
                )
                if "added_rating" in st.session_state:
                    st.success("Successfully added rating ✅")
                    st.session_state.pop("added_rating", None)
                elif selected_mode == "rated_movies":
                    st.warning("Movie already rated")
                elif selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
                    st.warning("Can't add ratings with Collaborative Filtering")

            with col_c2:
                if not "show_local_explanation" in st.session_state:
                    st.button(
                        label="📊 Show local recommendation explanation",
                        on_click=lambda: st.session_state.update(
                            {"show_local_explanation": True}),
                        disabled=(selected_mode == "rated_movies")
                    )
                else:
                    st.button(
                        label="📊 Hide local recommendation explanation",
                        on_click=lambda: st.session_state.pop("show_local_explanation", None)
                    )
        else:
            col_t1, col_t2, _ = st.columns(3)
            with col_t1:
                selected_rating = st.number_input(
                    f"Rating ⭐",
                    min_value=0.0,
                    max_value=5.0,
                    value=3.0,
                    step=1.0,
                    format="%.1f"  # 1 casa decimal
                )
            with col_t2:
                st.subheader(f"{'⭐'*int(selected_rating)}")

            # Add/Create new rating function
            col_c1, col_c2, _ = st.columns(3)
            with col_c1:
                def on_add_rating_callback(added_rating_dto: AddedRatingDTO):
                    # Call function to write rating to csv
                    add_data.add_rating_to_csv(added_rating_dto=added_rating_dto)
                    st.session_state["added_rating"] = added_rating_dto.rating
                    # Clear add_rating from memory and go back to movies page
                    for key in [
                        "add_rating", "selected_movie_id", "show_local_explanation", "selected_page", "selected_mode", "selected_genres", "search_query"]:
                        st.session_state.pop(key, None)

                st.button(
                    label="✅ Add Rating",
                    on_click=lambda: on_add_rating_callback(AddedRatingDTO(
                        userId=int(float(ui_movie_page_dto.selected_user)),
                        movieId=movie.get('movieId'),
                        rating=selected_rating,
                    ))
                )
            with col_c2:
                st.button(
                    label="❌ Cancel",
                    on_click=lambda: st.session_state.pop("add_rating", None)
                )

    st.markdown("---")

    if "show_local_explanation" in st.session_state:
        movie_local_explanation_container(
            selected_filtering_type=selected_filtering_type,
            ui_movie_cbf_local_explanation_container_dto=ui_movie_cbf_local_explanation_container_dto,
            ui_movie_cf_local_explanation_container_dto=ui_movie_cf_local_explanation_container_dto,
            ui_movie_hyb_local_explanation_container_dto=ui_movie_hyb_local_explanation_container_dto
        )


def movie_local_explanation_container(
        selected_filtering_type: FilteringType,
        ui_movie_cbf_local_explanation_container_dto: UiMovieCbfLocalExplanationContainerDto = None,
        ui_movie_cf_local_explanation_container_dto: UiMovieCfLocalExplanationContainerDto = None,
        ui_movie_hyb_local_explanation_container_dto: UiMovieHybLocalExplanationContainerDto = None
):
    """
    Container with the explanations about the selected movie recommendation

    :param selected_filtering_type:
    :param ui_movie_cbf_local_explanation_container_dto:
    :param ui_movie_cf_local_explanation_container_dto:
    :param ui_movie_hyb_local_explanation_container_dto:
    :return:
    """
    # Show Local SHAP explanation
    match selected_filtering_type.value:
        # Show Local explanation for CBF
        case FilteringType.CONTENT_BASED_FILTERING.value:
            # Clean selected movie for SHAP explanation
            selected_movie = (
                ui_movie_cbf_local_explanation_container_dto.predicted_movie_to_explain_df.copy()).reset_index()
            _, selected_movie = data_loader._clean_and_index_dfs(selected_movie, selected_movie,
                                                                 FilteringType.CONTENT_BASED_FILTERING)

            # Generate SHAP explanation for a single movie (local explainability)
            filtered_shap_values_local_explanation, filtered_feature_names_local_explanation = app_explainability.generate_shap_explanation_and_filter_out_zero_importance_features(
                prediction_model=ui_movie_cbf_local_explanation_container_dto.prediction_model,
                rated_movies_dataset=ui_movie_cbf_local_explanation_container_dto.rated_df,
                movies_to_explain_df=selected_movie,
                global_or_local="local",
                model_type=ui_movie_cbf_local_explanation_container_dto.selected_model_type
            )
            st.markdown("#### 💡 SHAP Explanation (Local)")

            # Charts side by side
            shap_local_chart_col1, shap_local_chart_col2 = st.columns(2)
            fig_bar_local, fig_pie_local = app_explainability.get_shap_bar_and_pie_chart_figures(
                filtered_shap_values_local_explanation, filtered_feature_names_local_explanation, "local")
            # Render charts
            col_a1, col_a2, _ = st.columns(3)

            ui_helpers.render_shap_explanation_charts(fig_bar_local, fig_pie_local)
            st.markdown("---")

        # Show Local explanation for CF
        case FilteringType.COLLABORATIVE_FILTERING.value:
            st.markdown("#### 💡 Neighbour Based Explanation (Local)")

            # Clean selected movie for SHAP explanation
            selected_movie = (
                ui_movie_cf_local_explanation_container_dto.predicted_movie_to_explain_df.copy()).reset_index()

            ratings_of_similar_users_df, explanation_text = app_explainability.generate_local_explanation_for_predictions_with_cf(
                selected_user_id=int(ui_movie_cf_local_explanation_container_dto.selected_user),
                model=ui_movie_cf_local_explanation_container_dto.prediction_model,
                rated_df_used_in_training=ui_movie_cf_local_explanation_container_dto.rated_df_used_in_training,
                user_index=ui_movie_cf_local_explanation_container_dto.user_index,
                predicted_movie_to_explain=selected_movie
            )
            if not ratings_of_similar_users_df.empty:
                #st.write(ratings_of_similar_users_df)
                pass
            #st.markdown("---")
            st.write(explanation_text)
            st.markdown("---")

        # Show Local explanation for Hybrid
        case FilteringType.HYBRID_FILTERING.value:
            st.markdown("#### 💡 Neighbour Based Explanation (Local)")

            selected_movie = (
                ui_movie_cbf_local_explanation_container_dto.predicted_movie_to_explain_df.copy()).reset_index()

            # st.write(ui_movie_hyb_local_explanation_container_dto.selected_user)
            # st.write(ui_movie_hyb_local_explanation_container_dto.rated_uncleaned_df)
            # st.write(selected_movie)

            ratings_of_similar_users_df_sorted = explainability.explain_local_predictions_for_hyb(
                selected_user_id=int(ui_movie_hyb_local_explanation_container_dto.selected_user),
                rated_uncleaned_df=ui_movie_hyb_local_explanation_container_dto.rated_uncleaned_df,
                predicted_movie_to_explain=selected_movie,
            )
            explanation_text: str = explainability.generate_neighbour_based_explanation_text(
                ratings_of_similar_users_df_sorted)

            if not ratings_of_similar_users_df_sorted.empty:
                #st.write(ratings_of_similar_users_df_sorted)
                pass
            #st.markdown("---")
            st.write(explanation_text)
            st.markdown("---")

            _, selected_movie = data_loader._clean_and_index_dfs(selected_movie, selected_movie,
                                                                 FilteringType.CONTENT_BASED_FILTERING)
            # Generate SHAP explanation for a single movie (local explainability)
            filtered_shap_values_local_explanation, filtered_feature_names_local_explanation = app_explainability.generate_shap_explanation_and_filter_out_zero_importance_features(
                prediction_model=ui_movie_cbf_local_explanation_container_dto.prediction_model,
                rated_movies_dataset=ui_movie_cbf_local_explanation_container_dto.rated_df,
                movies_to_explain_df=selected_movie,
                global_or_local="local",
                model_type=ui_movie_cbf_local_explanation_container_dto.selected_model_type
            )
            st.markdown("#### 💡 SHAP Explanation (Local)")

            # Charts side by side
            shap_local_chart_col1, shap_local_chart_col2 = st.columns(2)
            fig_bar_local, fig_pie_local = app_explainability.get_shap_bar_and_pie_chart_figures(
                filtered_shap_values_local_explanation, filtered_feature_names_local_explanation, "local")
            # Render charts
            col_a1, col_a2, _ = st.columns(3)

            ui_helpers.render_shap_explanation_charts(fig_bar_local, fig_pie_local)
            st.markdown("---")

            pass
        case _:
            raise ValueError(
                f"Unsupported filtering type: {selected_filtering_type}, type={type(selected_filtering_type)}")
