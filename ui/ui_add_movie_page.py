from datetime import date

import streamlit as st
from core import add_data  # Need clean and index function
from core.dtos.added_movie_dto import AddedMovieDTO
from ui.dtos.ui_add_movie_page_dto import UiAddMoviePageDto

if __name__ == '__main__':
    pass


def add_movie_page(
        ui_add_movie_page_dto: UiAddMoviePageDto
):
    """
    Creates an added_movie_dto and calls the function to save it in csv

    :param ui_add_movie_page_dto:
    :return:
    """
    st.markdown("#### ➕ Adding new Movie")

    # Layout: 2 columns
    col_a1, col_a2 = st.columns([1, 3])
    # Column 2: Details
    with col_a2:

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            movie_id = ui_add_movie_page_dto.next_unique_available_movie_id

            title = st.text_input(
                label="Title",
            )
            selected_genres = st.multiselect(
                label="Genres",
                options=ui_add_movie_page_dto.movie_genres,
            )
            release_date = st.date_input(
                label="Release date",
                value=date.today(),  # default value
                min_value=date(1950, 1, 1),
                max_value=date.today()
            )
        with col_d2:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                vote_average = st.number_input(
                    "TMDB Rating 🎬",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    format="%.1f"
                )
            with col_b2:
                popularity = st.number_input(
                    "Popularity 🍿",
                    min_value=0.0,
                    max_value=100.0,
                    value=5.0,
                    step=1.0,
                    format="%.1f"
                )
            overview = st.text_area(
                label="Overview",
                value="",  # initial value
                height=150,
                max_chars=1000
            )
        poster_url = st.text_input(
            "Poster's Url",
            value="",
            placeholder="https://example.com/poster.jpg"
        )

        st.subheader("🎬 Actions")
        col_c1, col_c2, _ = st.columns(3)
        with col_c1:
            def on_add_movie_callback(added_movie_dto: AddedMovieDTO):
                # Call function to write movie to csv
                add_data.add_movie_to_csv(added_movie_dto=added_movie_dto)
                st.session_state["added_movie"] = True
                # Clear add_movie from memory
                for key in [
                    "add_movie", "selected_page", "selected_mode", "selected_genres", "search_query"]:
                    st.session_state.pop(key, None)

            all_fields_filled = (
                    title
                    and selected_genres
                    and release_date
                    and vote_average is not None
                    and popularity is not None
                    and overview
                    and poster_url
            )
            st.button(
                label="✅ Add Movie",
                on_click=lambda: on_add_movie_callback(AddedMovieDTO(
                    movieId=movie_id,
                    title=title,
                    genres=selected_genres,
                    release_date=release_date,
                    vote_average=vote_average,
                    popularity=popularity,
                    overview=overview,
                    poster_url=poster_url)),
                disabled=not all_fields_filled
            )
        with col_c2:
            st.button(
                label="❌ Cancel",
                on_click=lambda: [st.session_state.pop(k, None) for k in
                          ("add_movie", "selected_page", "selected_mode", "selected_genres", "search_query")]
            )
    # Column 1: Poster
    with col_a1:
        if poster_url:
            try:
                st.image(poster_url, use_container_width=True)
            except Exception as e:
                st.image("https://www.envirochoice.com.au/Images/ProductImages/product-image-1.png")
        else:
            st.image("https://www.envirochoice.com.au/Images/ProductImages/product-image-1.png")
    st.markdown("---")
