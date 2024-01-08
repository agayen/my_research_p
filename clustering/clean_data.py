import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder

## if those are not there in your system install those
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

import pandas as pd
df = pd.read_json('data_set.json')

main_df = df.transpose()

main_df.head()

def clean_doc(document):
    # Define a pattern to match email addresses
    email_pattern = re.compile(r'\S+@\S+')

    # Use re.sub to replace email addresses with an empty string
    document = re.sub(email_pattern, '', document)

    sub_from_pattern = re.compile(r'^\s*(From:|Subject:).*$', flags=re.MULTILINE)

    # Use re.sub to remove lines matching the pattern
    document = re.sub(sub_from_pattern, '', document)

    # Remove full words starting with '@'
    document = re.sub(r'@\w+', '', document)

    # Remove special characters, URLs, and usernames
    document = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+|[^A-Za-z0-9]+', ' ', document)

    # Convert to lowercase
    document = document.lower()

    # Tokenize the document
    words = word_tokenize(document)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # POS tagging
    tagged_words = pos_tag(words)

    # Lemmatization using POS tags
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []

    for word, tag in tagged_words:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:
            lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    # Join the words back into a string
    cleaned_doc = ' '.join(lemmatized_words)

    return cleaned_doc

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return None  # Default to Noun if no match


print("clean data is starting...")
main_df['doc_text'] = main_df['doc_text'].apply(clean_doc)
breakpoint()
label_encoder = LabelEncoder()
main_df['number_label'] = label_encoder.fit_transform(main_df['doc_type'])
print("clean data is saving...")
main_df.to_csv('clean_data.csv', index=False)
print("clean data is saved...")

