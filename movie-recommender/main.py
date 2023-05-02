# import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("data/tmdb_movies_data.csv")

movies = df[['id','cast','director','genres','overview','original_title','keywords']]
movies.isnull().sum()
movies.dropna(inplace = True)

def convert(obj):
    s = list(obj)
    for i in range(len(s)):
        if s[i] == '|':
            s[i] = " "

    temp = "".join(s)
    ret = temp.split()
    return ret

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['cast'] = movies['cast'].apply(convert)
movies['director'] = movies['director'].apply(convert)

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['tags'] = movies['cast'] + movies['genres'] + movies['director'] + movies['overview'] + movies['keywords']

new_df = movies[['id', 'original_title', 'tags']]
new_df["tags"] = new_df["tags"].apply(lambda x : " ".join(x))
new_df["tags"] = new_df["tags"].apply(stem)

cv = CountVectorizer(max_features=10000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()
similarity = cosine_similarity(vectors)

print(len(similarity), len(movies))
pickle.dump(movies.to_dict(), open("pickle/movies_dict.pkl", "wb"))
pickle.dump(similarity, open("pickle/similarity.pkl", "wb"))