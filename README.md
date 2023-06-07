# Film Recommendation

## Description

The Film Recommendation is a Python code that utilizes vector similarities to analyze a given film description and compare it to a list of descriptions found in a file called "movies.txt". It then outputs the closest matching description from the list. 

The project aims to assist users in finding relevant movie recommendations based on the similarity of film descriptions. By leveraging the TF-IDF algorithm and cosine similarity, the system provides a convenient way to discover movies that align with a specific description or theme.

## Table of Contents

- [Description](#description)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Installation

To install and use the Film Recommendation locally, follow these steps:

1. Clone the repository to your local machine or download the `watch_next.py` file.

2. Ensure that you have the required dependencies installed. The code relies on the following libraries:
   - `numpy`: A library for numerical computing.
   - `scikit-learn`: A machine learning library that provides various tools for data preprocessing and feature extraction.

You can install the dependencies using `pip` by running the following command:
pip install numpy scikit-learn
   
3. Prepare the movie descriptions:
- Open file named "movies.txt".
- Add a movie description on a separate line.

4. Update the `description` variable in the `watch_next.py` script with the film description for which you want to find the closest match.

5. Run the script using a Python interpreter:
python watch_next.py

6. The output will be displayed on the console, showing the closest matching movie description from the "movies.txt" file.

## Usage

After installing the Film Recommendation, follow these steps to use it:

1. Prepare the movie descriptions by adding them to the "movies.txt" file. Each description should be on a separate line.

2. Update the `description` variable in the `watch_next.py` script with the film description you want to find a match for.

3. Run the script using a Python interpreter. The system will calculate the cosine similarities between the given description and the movie descriptions in the "movies.txt" file.

4. The script will output the title of the most similar movie description.

Below is an example of the code in action:

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def find_similar_movie(description):
 with open('movies.txt', 'r') as file:
     movies = file.readlines()

 vectorizer = TfidfVectorizer()
 X = vectorizer.fit_transform(movies)

 Y = vectorizer.transform([description])

 cosine_similarities = np.dot(X, Y.T).toarray().flatten()

 index = cosine_similarities.argmax()

 return movies[index].strip()

description = '''Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth,
the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can 
live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator.'''
similar_movie = find_similar_movie(description)
print(similar_movie) 
```

Replace the description variable with your desired film description, run the script, and the closest matching movie description will be displayed.

## Credits
This project was created by [Zainab Ismail] as part of [HyperionDev's Data Science tasks].
Feel free to contribute to the project or provide feedback. Enjoy discovering your next movie to watch!

