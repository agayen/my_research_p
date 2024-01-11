from __future__ import division
from cmath import sqrt
import nltk
import numpy
import pandas as pd
from collections import defaultdict
import json
from helper import get_main_results
from sklearn.decomposition import TruncatedSVD

def text_generator(data):
    for doc in data:
        yield doc

final_df = pd.read_csv('clean_data.csv')

documents = text_generator(final_df['doc_text'].apply(str).tolist())
labels = final_df['number_label']

texts = [nltk.Text(nltk.word_tokenize(raw)) for raw in documents]

# Empty list to hold text documents.
documents = []

# Iterate through the directory and build the collection of texts for NLTK.
dict1 = {}
dict1 = defaultdict(lambda: 0, dict1)
for i, text in enumerate(texts):
    tokens = nltk.word_tokenize(str(text))
    stemmed_tokens = nltk.Text(tokens)
    for x in tokens:
        dict1[x] += 1
    documents.append(stemmed_tokens)  # Update the texts list with the modified text

print("Prepared ", len(documents), " documents...")
print("They can be accessed using texts[0] - texts[" + str(len(documents)-1) + "]")

# Load the list of texts into a TextCollection object.
collection = nltk.TextCollection(documents)
print("Created a collection of", len(collection), "terms.")

# Get a list of unique terms
unique_terms = list(set(collection))
def cnt(x):
    return dict1[x]
unique_terms.sort(key=cnt, reverse=True)
print("Unique terms found: ", len(unique_terms))
newlist = []
for x in collection:
    if x in unique_terms[:3000]:
        newlist.append(x)

newcollection = nltk.TextCollection(newlist)

# Function to create a TF*IDF vector for one document.
def TFIDF(document):
    word_tfidf = []
    for word in unique_terms[:3000]:
        word_tfidf.append(newcollection.tf_idf(word, document))
    return word_tfidf

# And here we actually call the function and create our array of vectors.
document_term_matrix = [numpy.array(TFIDF(f)) for f in texts if len(f) != 0]

# Apply Latent Semantic Analysis (LSA) using Truncated SVD
n_topics = 20  # You can choose the number of topics
lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

# Display the document-topic matrix details
print("Document-Term Matrix for lsa created ..")

get_lsa_data = get_main_results(lsa_topic_matrix, labels)
print(get_lsa_data)

file_path = 'lsa_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_lsa_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')