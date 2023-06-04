#T21

'''This code uses vector similarities to analyze a film description 
and compare it to a list of descriptions found in a file called "movies.txt", 
and output the closest matching description. '''

import numpy as np # Import numpy library for numerical computing

from sklearn.feature_extraction.text import TfidfVectorizer
# Import TfidfVectorizer from sklearn's feature_extraction.text module for text processing and feature extraction using TF-IDF algorithm.

def find_similar_movie(description):
    # Read in the movies.txt file
    with open('movies.txt', 'r') as file:
        movies = file.readlines()

    # Use TF-IDF vectorizer to transform movie descriptions into feature vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(movies)

    # Transform the given description into a feature vector
    Y = vectorizer.transform([description])

    # Calculate cosine similarities between the given description and all the movies
    cosine_similarities = np.dot(X, Y.T).toarray().flatten()

    # Find the index of the movie with the highest cosine similarity
    index = cosine_similarities.argmax()

    # Return the title of the most similar movie
    return movies[index].strip()

description = '''Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth,
the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can 
live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator.'''
similar_movie = find_similar_movie(description)
print(similar_movie)