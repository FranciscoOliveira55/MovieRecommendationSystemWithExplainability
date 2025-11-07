import math
from enum import Enum

import pandas as pd
import streamlit as st
from fsspec.registry import default
from streamlit_option_menu import option_menu

from core import data_loader, external_api, model  # Need clean and index function
import ui.app_data_loader as app_data_loader
import ui.app_model as app_model
import ui.app_prediction as app_prediction
import ui.app_recommendation as app_recommendation
from configs import ShapConfig, LEARNINGCURVEConfig, FilteringType, ModelType
from ui.dtos.ui_add_movie_page_dto import UiAddMoviePageDto
from ui.dtos.ui_model_analytics_and_comparison_container_dto import UiModelAnalyticsAndComparisonContainerDto
from ui.dtos.ui_model_page_dto import UiModelPageDto
from ui.dtos.ui_movie_cbf_local_explanation_container_dto import UiMovieCbfLocalExplanationContainerDto
from ui.dtos.ui_movie_cf_local_explanation_container_dto import UiMovieCfLocalExplanationContainerDto
from ui.dtos.ui_movie_hyb_local_explanation_container_dto import UiMovieHybLocalExplanationContainerDto
from ui.dtos.ui_movie_page_dto import UiMoviePageDto
from ui.dtos.ui_movies_cbf_global_explanation_container_dto import UiMoviesCbfGlobalExplanationContainerDto
from ui.dtos.ui_movies_page_dto import UiMoviesPageDto
from ui.dtos.ui_user_page_dto import UiUserPageDto
from ui.ui_add_movie_page import add_movie_page
from ui.ui_model_page import model_page
from ui.ui_movie_page import movie_page
from ui.ui_movies_page import movies_page
from ui.ui_user_page import user_page


def run():
    """
    Run function (used to run the system)
    :return:
    """
    # --- Page configuration ---
    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        /* Fundo da página */
        .appview-container, .main, .block-container {
            background-image: url("https://cdn.vectorstock.com/i/500p/84/65/modern-white-geometric-background-vector-32028465.jpg");
            background-size: cover;
            background-repeat: repeat;
            background-attachment: fixed;
            background-position: center;
            background-color: transparent !important;
        }
        /* Remover cor de fundo branca do conteúdo */
        .css-18e3th9 {
            background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================================
    # ===================== # Initialize configs and  datasets  # =============================
    # =========================================================================================

    # Selected user
    # If selected user in session, keep that one, otherwise, grab the first one acording with configs (either first user or user with most ratings)
    if "selected_user" in st.session_state:
        selected_user: str = st.session_state["selected_user"]
    else:
        # Gets the CBF rated dataframe (with users 1-5, or top 5)
        cbf_rated_df_for_user_selection, _ = app_data_loader.initialize_dataframes(
            ratings_user_id=-1,
            clean_and_index_dfs=False,
            filtering_type=FilteringType.CONTENT_BASED_FILTERING
        )
        # Get list of unique users (1-5, or top 5)
        user_ids: pd.DataFrame = app_data_loader.get_user_ids_from_rated_df(cbf_rated_df_for_user_selection)
        # Select the first one
        selected_user: str = user_ids["userId"].iloc[0].astype(float).astype(int)
    # Save selected user in session
    st.session_state["selected_user"] = selected_user

    # Selected filtering type
    selected_filtering_type: FilteringType = st.session_state[
        "selected_filtering_type"] if "selected_filtering_type" in st.session_state else FilteringType.HYBRID_FILTERING  # FilteringType.CONTENT_BASED_FILTERING
    st.session_state["selected_filtering_type"] = selected_filtering_type

    # Selected model type
    selected_model_type: ModelType = st.session_state[
        "selected_model_type"] if "selected_model_type" in st.session_state else ModelType.NEURAL_NETWORK
    st.session_state["selected_model_type"] = selected_model_type

    # If a new item, interaction or user was just added, then reset transformed dfs from session and read new ones
    reseted_dfs: bool = ("added_movie" in st.session_state) or ("added_rating" in st.session_state) or (
                "added_user" in st.session_state)
    if reseted_dfs:
        for k in list(st.session_state.keys()):
            if "datasets" in k:
                st.session_state.pop(k, None)

    # Get uncleaned rated_df and unrated_df for selected user
    uncleaned_rated_df, uncleaned_unrated_df = app_data_loader.initialize_dataframes(
        ratings_user_id=int(float(selected_user)),
        clean_and_index_dfs=False,
        filtering_type=selected_filtering_type
    )

    # Count number of unique movies from the selected_user
    movies_rated_by_selected_user_df, number_of_unique_rated_movies_by_the_user = app_data_loader.get_user_ratings_from_rated_df(
        uncleaned_rated_df,
        selected_user
    )
    # Clean dfs if its CBF (if its CF, then model.prepare_training_data takes care of it)
    rated_df, unrated_df = app_data_loader.clean_and_index_user_ratings_df_and_full_unrated_df(
        user_ratings_df=uncleaned_rated_df,
        full_unrated_df=uncleaned_unrated_df,
        filtering_type=selected_filtering_type
    )

    # =========================================================================================
    # ============================ # Display nav bar  # =======================================
    # =========================================================================================
    # Navbar moderna
    current_display = option_menu(
        menu_title=None,
        options=["Recommended Movies", "Users", "Models"],
        icons=["film", "person", "cpu"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "color": "white",
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "15px 20px",
                "line-height": "30px"
            },
            "nav-link-selected": {"background-color": "#575757"},
        }
    )

    # Containers “colados” abaixo da navbar
    cols = st.columns(3, gap="small")  # gap pode ser "small", "medium", "large"

    with cols[0]:
        # st.write("📽️ **Info Movie:**")
        st.caption("Movie recommendation system")
        st.caption(f"Reseted dataframes: {str(reseted_dfs).lower()}")

    with cols[1]:
        # st.write("👤 **Info User:**")
        st.caption(f"Selected user: {int(float(selected_user))}")
        st.caption(
            f"User {int(float(selected_user))} has rated {int(number_of_unique_rated_movies_by_the_user)} unique movies.")

    with cols[2]:
        # st.write("🤖 **Info Model:**")
        st.caption(f"Selected filtering type: {selected_filtering_type.value.replace("_", " ").capitalize()}")
        st.caption(f"Selected model: {selected_model_type.value.replace("_", " ").capitalize()}")
    st.markdown("---")

    # =========================================================================================
    # ================== # Make model and predictions # =======================================
    # =========================================================================================
    # Create or read the model
    prediction_model, user_index, item_index = app_model.generate_model(
        rated_df=rated_df,
        model_type=selected_model_type,
        filtering_type=selected_filtering_type,
        user_id=int(float(selected_user))
    )

    # Make rating predictions for unrated movies
    predicted_rating_df = app_prediction.make_rating_predictions_for_unrated_movies_df(
        unrated_df=unrated_df,
        model=prediction_model,
        model_type=selected_model_type,
        filtering_type=selected_filtering_type,
        user_index=user_index,
        item_index=item_index
    )
    predicted_rating_df = predicted_rating_df.reset_index()


    # =========================================================================================
    # =================================== # Display  # ========================================
    # =========================================================================================

    match current_display:
        case "Recommended Movies":
            # Drop memory keys from User's and Model's pages
            for key in ("show_global_explanation",
                        "show_model_analytics_and_comparison"):
                st.session_state.pop(key, None)

            # =========================================================================================
            # ======================= # Transform data to display # ==================================
            # =========================================================================================
            movies_to_display_df: pd.DataFrame = pd.DataFrame()

            class ModeOptions(Enum):
                RECOMMENDED_MOVIES = "recommended_movies"
                RATED_MOVIES = "rated_movies"

            selected_mode: str = st.session_state.get("selected_mode", ModeOptions.RECOMMENDED_MOVIES.value)
            selected_genres: list[str] = st.session_state.get("selected_genres", [])
            search_query: str = st.session_state.get("search_query", "")

            # If collaborative filtering, grab movie features
            temp_features_of_rated_df, temp_features_of_unrated_df = app_data_loader.initialize_dataframes(
                ratings_user_id=int(float(selected_user)),
                clean_and_index_dfs=False,
                filtering_type=FilteringType.CONTENT_BASED_FILTERING
            )
            # if selected_filtering_type.value==FilteringType.COLLABORATIVE_FILTERING.value:
            # Drop useless columns
            _, temp_features_of_rated_df = data_loader._clean_and_index_dfs(
                rated_df=temp_features_of_rated_df,
                unrated_df=temp_features_of_rated_df,
                filtering_type=FilteringType.CONTENT_BASED_FILTERING
            )
            _, temp_features_of_unrated_df = data_loader._clean_and_index_dfs(
                rated_df=temp_features_of_unrated_df,
                unrated_df=temp_features_of_unrated_df,
                filtering_type=FilteringType.CONTENT_BASED_FILTERING
            )
            # Reset index (make movieId a column)
            temp_features_of_rated_df = temp_features_of_rated_df.reset_index()
            temp_features_of_unrated_df = temp_features_of_unrated_df.reset_index()
            # Pick either Rated or Recommended movies
            # st.write(f"Selected mode: {selected_mode}")
            match selected_mode:
                case ModeOptions.RECOMMENDED_MOVIES.value:
                    if selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
                        movies_to_display_df = pd.merge(
                            predicted_rating_df.copy(),
                            temp_features_of_unrated_df.copy(),
                            on="movieId",
                            how="inner"
                        )
                    else:
                        movies_to_display_df = pd.merge(
                            predicted_rating_df.copy(),
                            uncleaned_unrated_df[["movieId", "title"]].copy(),
                            on="movieId",
                            how="inner"
                        )
                case ModeOptions.RATED_MOVIES.value:
                    if selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value:
                        movies_to_display_df = pd.merge(
                            movies_rated_by_selected_user_df.copy(),
                            temp_features_of_rated_df.copy(),
                            on="movieId",
                            how="inner"
                        )
                    else:
                        movies_to_display_df = movies_rated_by_selected_user_df.copy()
                case _:
                    st.error(f"Mode of movies to display invalid: {selected_mode}, type={type(selected_mode)}")

            # Filter movies by genre
            if selected_genres and (not movies_to_display_df.empty):
                for genre in selected_genres:
                    movies_to_display_df: pd.DataFrame = movies_to_display_df[movies_to_display_df[genre] == 1]

            # Filter movies by title
            if search_query and (not movies_to_display_df.empty):
                # Breaks query in words
                words = search_query.strip().split()
                for w in words:
                    movies_to_display_df = movies_to_display_df[
                        movies_to_display_df["title"].str.contains(w, case=False)]

            # Add filtering bar
            show_filtering_bar: bool = (not "add_movie" in st.session_state) and (
                not "selected_movie_id" in st.session_state)
            if show_filtering_bar:
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    def on_change_mode_callback():
                        previously_selected_mode = st.session_state.get("_selected_mode", None)
                        st.session_state.pop("selected_page", None)
                        st.session_state["selected_mode"] = previously_selected_mode

                    st.selectbox(
                        label="Type",
                        options=[ModeOptions.RECOMMENDED_MOVIES.value, ModeOptions.RATED_MOVIES.value],
                        index=0,
                        key="_selected_mode",
                        format_func=lambda option: option.replace("_", " ").capitalize(),
                        on_change=on_change_mode_callback
                    )
                with col2:
                    def on_change_genres_callback():
                        previously_selected_genres = st.session_state.get("_selected_genres", None)
                        st.session_state.pop("selected_page", None)
                        st.session_state["selected_genres"] = previously_selected_genres

                    movie: pd.Series = temp_features_of_unrated_df.iloc[0]
                    movie_genres: list[str] = [col for col in movie.index if (movie[col] == 1 or movie[col] == 0) and (
                            col not in ["userId", "movieId"])]  # It is a movie's genre if its cell value is 1

                    st.multiselect(
                        label="Genres",
                        options=movie_genres,
                        key="_selected_genres",
                        on_change=on_change_genres_callback
                    )
                with col3:
                    def on_change_search_query_callback():
                        previous_search_query = st.session_state.get("_search_query", None)
                        st.session_state.pop("selected_page", None)
                        st.session_state["search_query"] = previous_search_query

                    st.text_input(
                        label="Search",
                        placeholder="Search for movie title",
                        key="_search_query",
                        on_change=on_change_search_query_callback
                    )
                st.divider()

            # If movies_to_display_df is empty, display "no results message"
            if movies_to_display_df.empty:
                st.markdown(
                    """
                    <h2 style='color: gray; opacity: 0.6; text-align: center;'>
                        No results found
                    </h2>
                    """,
                    unsafe_allow_html=True
                )
                return  # Doesnt go lower in the code

            # Order movies_to_display by PredictedRating/rating
            movies_to_display_df = movies_to_display_df.sort_values(
                by='PredictedRating' if selected_mode == ModeOptions.RECOMMENDED_MOVIES.value else "rating",
                ascending=False
            )

            # Split sorted movies_to_display in pages (20 movies per page)
            selected_page: int = st.session_state.get("selected_page", 1)
            movies_per_page: int = 20
            number_of_pages: int = math.ceil(len(movies_to_display_df) / movies_per_page)
            start_index: int = (selected_page - 1) * movies_per_page
            end_index: int = min(start_index + movies_per_page, len(movies_to_display_df))
            movies_to_display_in_current_page_df: pd.DataFrame = movies_to_display_df.iloc[start_index:end_index]

            # Add tmdb details to the movies_to_display[selected page]
            # st.dataframe(movies_to_display_in_current_page_df)

            movies_to_display_in_current_page_df = data_loader.add_links_to_movies_df(
                movies_to_display_in_current_page_df)

            # st.dataframe(movies_to_display_in_current_page_df)

            movies_to_display_with_tmdb_details_df = external_api.get_movies_details_from_tmdb(
                movies_to_display_in_current_page_df)

            #st.dataframe(predicted_rating_df)
            #st.dataframe(movies_to_display_df)

            # =========================================================================================
            # =================================== # Display  # ========================================
            # =========================================================================================

            if "add_movie" in st.session_state:
                # If the "add new movie" button was clicked
                # Grab "possible" movie genres
                movie: pd.Series = movies_to_display_with_tmdb_details_df.iloc[0]
                movie_genres: list[str] = [col for col in movie.index if (movie[col] == 1 or movie[col] == 0) and (
                        col not in ["userId", "movieId"])]  # It is a movie's genre if its cell value is 1

                ui_add_movie_page_dto = UiAddMoviePageDto(
                    movie_genres=movie_genres
                )
                add_movie_page(
                    ui_add_movie_page_dto=ui_add_movie_page_dto
                )

            elif "selected_movie_id" in st.session_state:
                # If a movie was selected
                selected_movie_id = st.session_state["selected_movie_id"]

                # Prepare UiMoviePageDto
                selected_movie_with_tmdb_details_df: pd.DataFrame = movies_to_display_with_tmdb_details_df.loc[
                    movies_to_display_with_tmdb_details_df["movieId"] == selected_movie_id]

                ui_movie_page_dto = UiMoviePageDto(
                    selected_user=str(selected_user),
                    selected_mode=selected_mode,
                    selected_movie_df=selected_movie_with_tmdb_details_df
                )

                # Prepare UiMovieCbfLocalExplanationContainerDto
                selected_movie_df: pd.DataFrame = predicted_rating_df.loc[
                    predicted_rating_df["movieId"] == selected_movie_id]
                ui_movie_cbf_local_explanation_container_dto = UiMovieCbfLocalExplanationContainerDto(
                    selected_model_type=selected_model_type,
                    prediction_model=prediction_model,
                    rated_df=rated_df,
                    predicted_movie_to_explain_df=selected_movie_df
                ) if ((selected_filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value) or
                      (selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value)) else None

                # Prepare UiMovieHybLocalExplanationContainerDto
                ui_movie_hyb_local_explanation_container_dto = UiMovieHybLocalExplanationContainerDto(
                    selected_user=selected_user,
                    rated_uncleaned_df=uncleaned_rated_df,
                    predicted_movie_to_explain=selected_movie_df
                ) if selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value else None

                # Prepare UiMovieCfLocalExplanationContainerDto
                ui_movie_cf_local_explanation_container_dto = UiMovieCfLocalExplanationContainerDto(
                    selected_user=selected_user,
                    prediction_model=prediction_model,
                    rated_df_used_in_training=rated_df,
                    user_index=user_index,
                    predicted_movie_to_explain_df=selected_movie_df
                ) if selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value else None

                movie_page(
                    ui_movie_page_dto=ui_movie_page_dto,
                    selected_filtering_type=selected_filtering_type,
                    ui_movie_cbf_local_explanation_container_dto=ui_movie_cbf_local_explanation_container_dto,
                    ui_movie_cf_local_explanation_container_dto=ui_movie_cf_local_explanation_container_dto,
                    ui_movie_hyb_local_explanation_container_dto=ui_movie_hyb_local_explanation_container_dto
                )
            else:
                # Show the list of recommended movies
                ui_movies_page_dto = UiMoviesPageDto(
                    selected_user=str(selected_user),
                    selected_filtering_type=selected_filtering_type,
                    selected_model_type=selected_model_type,
                    selected_mode=selected_mode,
                    predicted_rating_df=movies_to_display_with_tmdb_details_df
                )
                movies_page(ui_movies_page_dto=ui_movies_page_dto)

                # =========================================================================================
                # ============================== # Page Navigation  # =====================================
                # =========================================================================================
                st.markdown("---")
                window_size = 5  # quantas opções mostrar
                half_window = window_size // 2
                start_page = max(1, selected_page - half_window)
                end_page = min(number_of_pages, start_page + window_size - 1)
                start_page = max(1, end_page - window_size + 1)  # ajusta se estiver perto do fim
                page_options = list(range(start_page, end_page + 1))

                cols_f1, cols_f2, _ = st.columns([1, 2, 1])
                cols = st.columns(len(page_options))
                for i, col in enumerate(cols):
                    page_num = page_options[i]
                    # Gives special attention to current page
                    label = f"▶ Page {page_num} ◀" if page_num == selected_page else str(page_num)
                    col.button(
                        label=label,
                        on_click=lambda p=page_num: st.session_state.update({"selected_page": p}),
                        use_container_width=True
                    )

        case "Users":
            # Drop memory keys from Movie's and Model's pages
            for key in ("selected_movie_id",
                        "show_local_explanation",
                        "show_model_analytics_and_comparison",
                        "add_movie",
                        "add_rating",
                        "selected_page", "selected_mode", "selected_genres", "search_query"):
                st.session_state.pop(key, None)

            ui_user_page_dto = UiUserPageDto(
                selected_user=selected_user,
                number_of_unique_rated_movies_by_the_user=number_of_unique_rated_movies_by_the_user,
                movies_rated_by_selected_user_df=movies_rated_by_selected_user_df
            )
            ui_movies_cbf_global_explanation_container_dto = UiMoviesCbfGlobalExplanationContainerDto(
                selected_model_type=selected_model_type,
                prediction_model=prediction_model,
                rated_df=rated_df,
                predicted_movies_to_explain_df=predicted_rating_df.sample(
                    min(ShapConfig.GLOBAL_EXPLANATION_BATCH_SIZE, len(predicted_rating_df)),
                    random_state=ShapConfig.GLOBAL_EXPLANATION_BATCH_RANDOM_STATE)
            ) if ((selected_filtering_type.value == FilteringType.CONTENT_BASED_FILTERING.value) or
                  (selected_filtering_type.value == FilteringType.HYBRID_FILTERING.value)) else None
            user_page(
                ui_user_page_dto=ui_user_page_dto,
                selected_filtering_type=selected_filtering_type,
                ui_movies_cbf_global_explanation_container_dto=ui_movies_cbf_global_explanation_container_dto
            )
        case "Models":
            # Drop memory keys from Movie's and User's pages
            for key in ("selected_movie_id",
                        "show_local_explanation",
                        "show_global_explanation",
                        "add_movie",
                        "add_rating",
                        "add_user",
                        "selected_page", "selected_mode", "selected_genres", "search_query"):
                st.session_state.pop(key, None)

            ui_model_page_dto = UiModelPageDto(
                selected_user=selected_user,
                selected_filtering_type=selected_filtering_type,
                selected_model_type=selected_model_type
            )
            ui_model_analytics_and_comparison_container_dto = UiModelAnalyticsAndComparisonContainerDto(
                selected_user=selected_user,
                selected_filtering_type=selected_filtering_type,
                selected_model_type=selected_model_type,
                rated_df=rated_df,
                prediction_model=prediction_model
            )
            model_page(
                ui_model_page_dto=ui_model_page_dto,
                ui_model_analytics_and_comparison_container_dto=ui_model_analytics_and_comparison_container_dto
            )
        case _:
            raise Exception(f"Unknown page: {current_display}")


if __name__ == '__main__':
    run()
