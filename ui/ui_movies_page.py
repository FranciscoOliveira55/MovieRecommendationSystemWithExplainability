import math

import pandas as pd
import streamlit as st

from configs import FilteringType
from core import data_loader, external_api
from ui import ui_helpers, app_data_loader, app_prediction, app_recommendation, app_explainability
from ui.dtos.ui_movies_page_dto import UiMoviesPageDto

if __name__ == '__main__':
    pass


def movies_page(ui_movies_page_dto: UiMoviesPageDto):
    """
    Page with all the recommendation movies displayed

    :param ui_movies_page_dto:
    :return:
    """
    # st.dataframe(ui_movies_page_dto.predicted_rating_df.head())

    col1, col2, col3 = st.columns([3.9, 1.1, 1])

    with col1:
        selected_mode = ui_movies_page_dto.selected_mode
        match selected_mode:
            case "recommended_movies":
                st.subheader(f"Top Recommended Movies for User {int(float(ui_movies_page_dto.selected_user))}")
            case "rated_movies":
                st.subheader(f"Top Rated Movies by User {int(float(ui_movies_page_dto.selected_user))}")
            case _:
                st.error(f"Mode of movie to display invalid: {selected_mode}, type={type(selected_mode)}")
    with col2:
        if "added_movie" in st.session_state:
            st.success("Successfully added movie ✅")
            st.session_state.pop("added_movie", None)
        if "added_rating" in st.session_state:
            st.success("Successfully added rating ✅")
            st.session_state.pop("added_rating", None)
    with col3:
        disable_add_button:bool = (ui_movies_page_dto.selected_filtering_type.value == FilteringType.COLLABORATIVE_FILTERING.value)
        st.button(
            label="➕ Add new Movie",
            key="add_movie_btn",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                {"add_movie": True}),
            disabled= disable_add_button
        )
        if disable_add_button:
            st.warning("Can't add movies with Collaborative Filtering")

    # Grab recommended movies
    recommended_movies_with_tmdb_details_df = ui_movies_page_dto.predicted_rating_df

    movies_per_row = 5
    num_movies = len(recommended_movies_with_tmdb_details_df)
    num_rows = math.ceil(num_movies / movies_per_row)

    # For each row of movies
    for row_idx in range(num_rows):
        start_idx = row_idx * movies_per_row
        end_idx = min(start_idx + movies_per_row, num_movies)
        cols = st.columns(end_idx - start_idx)  # Creates columns for the movies of the current row

        # For each movie in the row
        for i, idx in enumerate(range(start_idx, end_idx)):
            col = cols[i]
            with (col):
                target_movie = recommended_movies_with_tmdb_details_df.iloc[[idx]]  # In 1 row df
                # If target movie is not empty, display it
                if not target_movie.empty:
                    # Grab movie details
                    target_movie_id = target_movie.iloc[0]['movieId']
                    title = target_movie.iloc[0].get('title', 'No Title')
                    poster_url = target_movie.iloc[0]['poster_url']
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image("https://www.envirochoice.com.au/Images/ProductImages/product-image-1.png",
                                 use_container_width=True)

                    st.button(
                        label=f"{title}",
                        key=f"movie_{target_movie_id}",
                        use_container_width=True,
                        on_click=lambda movie_id=target_movie_id: st.session_state.update(
                            {"selected_movie_id": movie_id})
                    )
                else:
                    st.warning("Movie not found.")
