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

texts = [nltk.Text(nltk.word_tokenize(raw)) for raw in documents]

#Empty list to hold text documents.
documents = []
stopwords = set(nltk.corpus.stopwords.words('english'))

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if item not in stopwords:
            stemmed.append(stemmer.stem(item))    
    return stemmed

# Iterate through the directory and build the collection of texts for NLTK.
dict1 = {}
dict1 = defaultdict(lambda: 0, dict1)
for i, text in enumerate(texts):
    tokens = nltk.word_tokenize(str(text))
    # tokens = stem_tokens(tokens,  nltk.PorterStemmer())
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
vectors_tfidf = [numpy.array(TFIDF(f)) for f in texts if len(f) != 0]
print("TF-IDF Vectors created.")

get_tf_idf_data = get_main_results(vectors_tfidf, labels)
print(get_tf_idf_data)

file_path = 'tf_idf_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_tf_idf_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')