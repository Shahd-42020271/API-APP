from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Load dataset
pd.options.display.max_colwidth = 250
df = pd.read_csv("C:\\Users\\Yasmine\\Desktop\\New folder (3)\\SpecialtyDescriptions.csv", usecols=["Specialty", "sentence"])

# Preprocessing functions
STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200

PATTERN_S = re.compile("\'s")
PATTERN_RN = re.compile("\\r\\n")
PATTERN_PUNC = re.compile(r"[^\w\s]")

def clean_text(text):
    text = text.lower()
    text = re.sub(PATTERN_S, ' ', text)
    text = re.sub(PATTERN_RN, ' ', text)
    text = re.sub(PATTERN_PUNC, ' ', text)
    return text

def tokenizer(sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
    if lemmatize:
        stemmer = WordNetLemmatizer()
        tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
    else:
        tokens = [w for w in word_tokenize(sentence)]
    token = [w for w in tokens if (len(w) > min_words and len(w) < max_words and w not in stopwords)]
    return tokens

def clean_sentences(df):
    df.dropna(inplace=True)
    df['clean_sentence'] = df['sentence'].apply(clean_text)
    df['tok_lem_sentence'] = df['clean_sentence'].apply(
        lambda x: tokenizer(x, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True))
    return df

df = clean_sentences(df)

# Utility function
def extract_best_indices(m, topk, mask=None):
    if len(m.shape) > 1:
        cos_sim = np.mean(m, axis=0)
    else:
        cos_sim = m
    index = np.argsort(cos_sim)[::-1]
    if mask is not None:
        assert mask.shape == m.shape
        mask = mask[index]
    else:
        mask = np.ones(len(cos_sim))
    mask = np.logical_or(cos_sim[index] != 0, mask)
    best_index = index[mask][:topk]
    return best_index

# TF-IDF  
token_stop = tokenizer(' '.join(STOPWORDS), lemmatize=False)
vectorizer = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer)
tfidf_mat = vectorizer.fit_transform(df['sentence'].values)

# Prediction function
def get_recommendations_tfidf(sentence, tfidf_mat):
    tokens = [str(tok) for tok in tokenizer(sentence)]
    vec = vectorizer.transform(tokens)
    mat = cosine_similarity(vec, tfidf_mat)
    best_index = extract_best_indices(mat, topk=3)
    return best_index

@app.route('/predict', methods=['POST'])
def predict_specialties():
    data = request.get_json()
    description = data['description']
    best_index = get_recommendations_tfidf(description, tfidf_mat)
    specialties = df['Specialty'].iloc[best_index].tolist()
    return jsonify({'specialties': specialties})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        description = request.form['description']
        best_index = get_recommendations_tfidf(description, tfidf_mat)
        specialties = df['Specialty'].iloc[best_index].tolist()
        return render_template('index.html', specialties=specialties)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
