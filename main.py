import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance

def load_data():
    """
    Loads the data from the CSV files and returns the datasets.
    """
    netflix_df = pd.read_csv("n_movies.csv")
    imdb_df = pd.read_csv("imdb_top_1000.csv")
    return netflix_df, imdb_df

def preprocess_dataframes(netflix_df, imdb_df):
    """
    Preprocesses the IMDB and Netflix datasets.
    """
    netflix_df = netflix_df.rename(columns={
        'title': 'Title',
        'certificate': 'Certificate',
        'genre': 'Genre',
        'rating': 'Rating',
        'description': 'Overview',
        'stars': 'Stars',
        'votes': 'No_of_Votes',
        'Released_Year': 'Released_Year',
        'duration_in_min': 'Runtime'
    })

    imdb_df = imdb_df.rename(columns={
        'Series_Title': 'Title',
        'IMDB_Rating': 'Rating',
        'No_of_Votes': 'No_of_Votes',
        'Overview': 'Overview',
        'Runtime': 'Runtime',
        'stars': 'Stars'
    })

    netflix_df['Genre'] = netflix_df['Genre'].str.lower().str.split(', ')
    imdb_df['Genre'] = imdb_df['Genre'].str.lower().str.split(', ')

    combined_df = pd.concat([netflix_df, imdb_df], ignore_index=True)
    combined_df['Genre'] = combined_df['Genre'].apply(lambda x: x if isinstance(x, list) else [])
    
    return combined_df

def cluster_movies_by_genre(combined_df):
    """
    Cluster movies by genre and add the cluster labels to the dataframe.
    """
    genres_encoded = combined_df['Genre'].explode().str.get_dummies().groupby(level=0).sum()
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    combined_df['Cluster'] = kmeans.fit_predict(genres_encoded)
    return combined_df, genres_encoded

def recommend_movies_nearest_updated_cosine(movie_title, genres_encoded, combined_df, num_recommendations=5):
    """
    Recommends movies based on the provided movie title using cosine similarity.
    """
    if movie_title not in combined_df['Title'].values:
        return []

    movie_data = combined_df[combined_df['Title'] == movie_title]
    movie_cluster = movie_data['Cluster'].iloc[0]
    movie_vector = genres_encoded.loc[movie_data.index].values[0]

    cluster_movies = combined_df[combined_df['Cluster'] == movie_cluster]
    cluster_movies_vectors = genres_encoded.loc[cluster_movies.index]

    similarities = cluster_movies_vectors.apply(lambda row: distance.cosine(row, movie_vector), axis=1)
    nearest_movies = similarities.nsmallest(num_recommendations).index

    recommended_movie_titles = combined_df.loc[nearest_movies]['Title'].tolist()
    if movie_title in recommended_movie_titles:
        recommended_movie_titles.remove(movie_title)

    return recommended_movie_titles
