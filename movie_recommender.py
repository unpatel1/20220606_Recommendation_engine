# Building a Movie Recommendation Engine (content based) | Machine Learning Projects
# https://www.youtube.com/watch?v=XoTwndOgXBM
# 20220606

# Importing the necessary libraries

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions #######

def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
####################################################


# Step 1: Read CSV File

data_dir_path = os.path.dirname(__file__)
data_url = os.path.join(data_dir_path, 'movie_dataset.csv')
df = pd.read_csv(data_url)
#print(df.head())
#print(df.columns)

# Step 2: Select Features

features = ['keywords','cast','genres','director']

# Taking care of NaN -- replacing with ''
for feature in features:
	df[features] = df[features].fillna('')

# Step 3: Create a column in DF which combines all selected features

def combine_features(row):
	try:
		return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
	except:
		print("Error: ", row)

df["combined_features"] = df.apply(combine_features, axis = 1)
# print(df["combined_features"].head())

# Step 4: Create count matrix from this new combined column

# CountVectorizer is a class so we have to initialize a "cv" object. 
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Step 5: Compute the Cosine Similarity based on the count_matrix

# cosine_similarity is not a class but a method so we do not need to initialize an object.
cosine_similarity_scores = cosine_similarity(count_matrix)

movie_user_likes = "Avatar"

# Step 6: Get index of this movie from its title

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_similarity_scores[movie_index]))

# Step 7: Get a list of similar movies in descending order of similarity score

sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True)
# print(sorted_similar_movies[:50])

# Step 8: Print titles of first 50 movies

i = 0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i += 1
	if i > 50:
		break

# *** END ***
