import streamlit as st
from main import load_data, preprocess_dataframes, cluster_movies_by_genre, recommend_movies_nearest_updated_cosine
from tmdbv3api import TMDb, Movie
import random
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import os

# Initialize the TMDb object with your API key
tmdb = TMDb()
tmdb.api_key = "933168374b24ebd0de4b9a15e213132c"  # Your API Key
movie_search = Movie()

BASE_TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500/"
DEFAULT_POSTER = "https://via.placeholder.com/500x750.png?text=No+Poster+Available"

def fetch_movie_details(movie_title):
    movie_id = None
    try:
        search_results = movie_search.search(movie_title)

        if not search_results:
            return {
                "id": movie_id,
                "poster": DEFAULT_POSTER,
                "release_date": "Unknown",
                "rating": "N/A",
                "overview": "No overview available",
                "genres": "Unknown"
            }

        movie_id = search_results[0].id
        movie_details = movie_search.details(movie_id)

        return {
            "id": movie_id,
            "poster": BASE_TMDB_IMAGE_URL + movie_details.poster_path if movie_details.poster_path else DEFAULT_POSTER,
            "release_date": movie_details.release_date if hasattr(movie_details, 'release_date') else "Unknown",
            "rating": str(movie_details.vote_average) if hasattr(movie_details, 'vote_average') else "N/A",
            "overview": movie_details.overview if hasattr(movie_details, 'overview') else "No overview available",
            "genres": ", ".join([genre['name'] for genre in movie_details.genres]) if hasattr(movie_details, 'genres') else "Unknown"
        }

    except Exception as e:
        print(f"An error occurred while fetching details for {movie_title}: {e}")
        return {
            "id": movie_id,
            "poster": DEFAULT_POSTER,
            "release_date": "Unknown",
            "rating": "N/A",
            "overview": "No overview available",
            "genres": "Unknown"
        }

def movie_recommendation_page():
    st.title("🎬 Movie Recommendation System")

    netflix_df, imdb_df = load_data()
    combined_df = preprocess_dataframes(netflix_df, imdb_df)
    combined_df, genres_encoded = cluster_movies_by_genre(combined_df)

    st.write("Discover movies similar to your favorites!")

    auto_trigger = False

    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'potential_matches' not in st.session_state:
        st.session_state.potential_matches = []
    if 'movie_input' not in st.session_state:
        st.session_state.movie_input = ""
    if 'reset_triggered' not in st.session_state:
        st.session_state.reset_triggered = False
    if 'surprise_triggered' not in st.session_state:
        st.session_state.surprise_triggered = False

    default_value = "" if st.session_state.reset_triggered else st.session_state.movie_input

    new_input = st.text_input("🔍 Masukan Film Favorit Kamu:", value=default_value, key="movie_input")

    if new_input != st.session_state.movie_input:
        st.session_state.movie_input = new_input
        st.session_state.reset_triggered = False

    movie_title = st.session_state.movie_input

    if movie_title:
        exact_match = combined_df[combined_df['Title'] == movie_title]

        if exact_match.empty:
            st.session_state.potential_matches = combined_df[combined_df['Title'].str.contains(movie_title, case=False, na=False)]['Title'].tolist()

            if st.session_state.potential_matches:
                selected_title = st.selectbox("Did you mean one of these?", st.session_state.potential_matches)
                if selected_title:
                    movie_title = selected_title
                    auto_trigger = True

    col1, col2, col3 = st.columns(3)

    if col1.button('Get Recommendations', key='btn_get_recommendations') or auto_trigger:
        with st.spinner('Fetching recommendations...'):
            st.session_state.recommendations = recommend_movies_nearest_updated_cosine(
                movie_title, genres_encoded=genres_encoded, combined_df=combined_df
            )
        display_recommendations(st.session_state.recommendations)

    if col2.button('Berikan Aku Saran!', key='btn_surprise_me'):
        st.session_state.recommendations = []
        st.session_state.potential_matches = []
        st.session_state.reset_triggered = True

        movie_title = random.choice(combined_df['Title'].tolist())
        with st.spinner('Fetching recommendations...'):
            st.session_state.recommendations = recommend_movies_nearest_updated_cosine(
                movie_title, genres_encoded=genres_encoded, combined_df=combined_df
            )
        display_recommendations(st.session_state.recommendations)

    if col3.button("Reset", key="btn_reset"):
        st.session_state.recommendations = []
        st.session_state.potential_matches = []
        st.session_state.reset_triggered = True
        st.experimental_rerun()

def display_recommendations(recommendations):
    if recommendations:
        st.subheader("Recommended Movies:")

        cols = st.columns(len(recommendations))

        for idx, movie in enumerate(recommendations):
            movie_details = fetch_movie_details(movie)

            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div style="overflow: hidden; border-radius: 10px;">
                            <img src="{movie_details['poster']}" alt="{movie}" style="width: 150px; height: 225px; object-fit: cover;">
                        </div>
                        <a href="https://www.themoviedb.org/movie/{movie_details['id']}" target="_blank">
                            <div class="movie-title">{movie}</div>
                        </a>
                        <p class="movie-info">{movie_details['release_date']} | {movie_details['genres']} | Rating: {movie_details['rating']}</p>
                        <details>
                            <summary>Overview</summary>
                            {movie_details['overview']}
                        </details>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.error("Couldn't find any recommendations for that movie.")

def wordcloud():
    st.title("WordCloud dan Diagram Batang dari CSV")

    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV Anda:", type=["csv"])
    if uploaded_file is not None:
        # Baca file CSV
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File berhasil diunggah!")
            st.write("Pratinjau Data:")
            st.dataframe(data.head())

            # Pilih kolom
            columns = data.columns.tolist()
            column_name = st.selectbox("Pilih kolom teks untuk WordCloud:", columns)

            if column_name:
                # Gabungkan semua teks dari kolom yang dipilih
                text = " ".join(data[column_name].dropna().astype(str))

                # Generate WordCloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white', 
                    max_words=200, 
                    colormap='rainbow'
                ).generate(text)

                # Tampilkan WordCloud
                st.subheader("WordCloud:")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)

                # Hitung frekuensi kata untuk diagram batang
                words = text.split()
                word_freq = Counter(words)
                common_words = word_freq.most_common(10)

                # Plot diagram batang
                words, counts = zip(*common_words)
                st.subheader("Diagram Batang - Top 10 Kata:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(words, counts, color='steelblue')
                ax.set_xlabel('Frequency')
                ax.set_title(f'Top 10 Words in {column_name}')
                ax.invert_yaxis()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

def main():
    st.sidebar.title("Tugas Besar Big Data Kelompok 9")
    pages = {
        "Movie Recommendations": movie_recommendation_page,
        "Word Cloud": wordcloud,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    page_function = pages[selection]
    page_function()

if __name__ == "__main__":
    main()
