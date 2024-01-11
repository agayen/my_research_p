from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from helper import get_main_results
import json

def text_generator(data):
    for doc in data:
        yield doc

final_df = pd.read_csv('clean_data.csv')

documents = text_generator(final_df['doc_text'].values.astype('S'))
labels = final_df['number_label']

# Create a TfidfVectorizer
max_words = 3000
vectorizer = TfidfVectorizer(max_features=max_words)

# Fit and transform the documents into TF-IDF vectors
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert the TF-IDF matrix to a dense array for better readability
vectors_tfidf = tfidf_matrix.toarray()

# Print the TF-IDF matrix
print("TF-IDF Matrix:")
print(vectors_tfidf)

print("TF-IDF Vectors created.")

get_tf_idf_data = get_main_results(vectors_tfidf, labels)
print(get_tf_idf_data)

file_path = 'tf_idf_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_tf_idf_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')