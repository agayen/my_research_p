from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from helper import get_main_results
import json

final_df = pd.read_csv('clean_data.csv')
# Sample collection of documents
documents = final_df['doc_text'].values.astype('U')
labels = final_df['number_label']
# Step 1: Convert documents to a bag-of-words (BoW) representation
vectorizer = CountVectorizer(max_features=3000)
bow_matrix = vectorizer.fit_transform(documents)

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