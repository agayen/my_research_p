import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import json
from helper import get_main_results

final_df = pd.read_csv('clean_data.csv')
# final_df = final_df.head()
documents = final_df['doc_text'].values.astype('U')
labels = final_df['number_label']

# Create a document-term matrix using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
document_term_matrix = vectorizer.fit_transform(documents)

# Apply Latent Semantic Analysis (LSA) using Truncated SVD
n_topics = 20  # You can choose the number of topics
lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

# Display the top words for each topic
terms = vectorizer.get_feature_names_out()
for i, topic in enumerate(lsa_model.components_):
    top_keywords_idx = topic.argsort()[-5:][::-1]
    top_keywords = [terms[idx] for idx in top_keywords_idx]
    print(f"Topic {i + 1}: {', '.join(top_keywords)}")

# Display the document-topic matrix details
print("Document-Term Matrix:")
print(document_term_matrix.shape)
print("\nLSA Topic Matrix:")
print(lsa_topic_matrix.shape)
print("\nExplained Variance Ratio:")
print(lsa_model.explained_variance_ratio_)


get_lsa_data = get_main_results(lsa_topic_matrix, labels)
print(get_lsa_data)

file_path = 'las_results.json'
with open(file_path, 'w') as json_file:
    json.dump(get_lsa_data, json_file)

print(f'The dictionary has been saved as the result in {file_path}')