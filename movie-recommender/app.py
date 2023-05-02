import streamlit as st
import pickle
import pandas as pd
import requests
from key import API_KEY

movies_dict = pickle.load(open('pickle/movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('pickle/similarity.pkl','rb'))

def fetch_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US')
    data = response.json()
    
    if 'poster_path' in data.keys():
        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    return "https://image.tmdb.org/t/p/w500/wwemzKWzjKYJFfCeiB57q3r4Bcm.png"

def recommend(movie):
    movie_index = 0
    for i, row in movies.iterrows():
        if row['original_title'] == movie:
            break
        movie_index += 1
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x : x[1])[1:6]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].original_title)
        recommended_movies_poster.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_movies_poster

st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    'Select a movie to recommend',
    movies['original_title'].values
)

if st.button("Recommend"):
    name, posters = recommend(selected_movie_name)

    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(name[i])
            st.image(posters[i])