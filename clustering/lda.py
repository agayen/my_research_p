from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from helper import get_main_results
import json
import nltk
from collections import defaultdict

def text_generator(data):
    for doc in data:
        yield doc

final_df = pd.read_csv('clean_data.csv')

documents = text_generator(final_df['doc_text'].apply(str).tolist())
labels = final_df['number_label']

# Step 1: Convert documents to a bag-of-words (BoW) representation


texts = [nltk.Text(nltk.word_tokenize(raw)) for raw in documents]

#Empty list to hold text documents.
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

# Function to create a Bag-of-Words vector for one document.
def BOW(document, vocabulary):
    text_string = ' '.join(document)

    # Create a CountVectorizer with the specified vocabulary
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    bow_vector = vectorizer.fit_transform([text_string]).toarray().flatten()
    return bow_vector

# And here we call the function and create our array of vectors.
vocabulary_bow = list(set(newlist))  # Use the unique terms for BOW
bow_matrix = [BOW(f, vocabulary_bow) for f in texts if len(f) != 0]

# Step 2: Apply Latent Dirichlet Allocation (LDA)
num_topics = 20  # Adjust the number of topics based on your needs
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_matrix = lda.fit_transform(bow_matrix)

# Print the document-topic matrix
print("\nDocument-Topic Matrix:")
print(lda_matrix)

get_lsa_data = get_main_results(lda_matrix, labels)
print(get_lsa_data)

file_path = 'lda_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_lsa_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')