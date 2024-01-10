import nltk
from collections import defaultdict
import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from helper import get_main_results
import json

final_df = pd.read_csv('clean_data.csv')

documents = final_df['doc_text'].values.astype('U')
labels = final_df['number_label']
max_words = 3000
vectorizer = CountVectorizer(max_features=max_words)

vectors_bow = vectorizer.fit_transform(documents)
vectors_bow = numpy.array(vectors_bow.todense())

print("Bag-of-Words Vectors created.")

get_bow_data = get_main_results(vectors_bow, labels)
print(get_bow_data)

file_path = 'bow_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_bow_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')