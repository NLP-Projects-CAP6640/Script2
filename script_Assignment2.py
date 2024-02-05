import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import json
nltk.download('stopwords')

# 1. Data Collection --------------------------------------------------------------------------------

# Loading json
with open('Science_Technology_News.json', encoding='utf-8') as f:
    data = json.load(f)

# Main dataframe
df = pd.json_normalize(data)

# Printing Main dataframe structure
print(df.info())

# 2. Information Extraction -----------------------------------------------------------------

# Extracting relevant fields
df = df[['title', 'pubDate', 'creator', 'content']].copy()

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text):
    if text:
        tokens = tokenizer.tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words]
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        return " ".join(stemmed_tokens)
    return ""

# Applying preprocessing to content field
df['processed_content'] = df['content'].apply(preprocess)

# Printing Main dataframe structure
print(df.info())

# Printing preprocessed content
print(df['processed_content'].head())

# Topic Modeling -----------------------------------------------------------------

# Analysis -----------------------------------------------------------------
