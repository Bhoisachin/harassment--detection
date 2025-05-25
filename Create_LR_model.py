import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import re, string
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_json('Dataset.json')

# Drop unnecessary columns
df.drop(['extras'], axis=1, inplace=True)

# Convert annotations to binary labels
df['annotation'] = df['annotation'].apply(lambda x: 1 if x['label'][0] == '1' else 0)

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords and punctuation removal
stop = stopwords.words('english')
regex = re.compile('[%s]' % re.escape(string.punctuation))

def preprocess_text(text):
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop])
    # Remove punctuation
    text = regex.sub('', text)
    # Tokenize and stem
    porter_stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([porter_stemmer.stem(word) for word in tokens if not word.isdigit()])
    return text

# Apply preprocessing
df['content'] = df['content'].apply(preprocess_text)

# Initialize TfidfVectorizer
tfIdfVectorizer = TfidfVectorizer(use_idf=True, sublinear_tf=True)
tfIdf = tfIdfVectorizer.fit_transform(df['content'])

# Save the TfidfVectorizer
dump(tfIdfVectorizer, 'tfidf_vectorizer.pkl')

# Prepare data for model training
X = tfIdf.toarray()
y = np.array(df['annotation'])

# Handle class imbalance
oversample = RandomOverSampler(sampling_strategy='not majority')
X_over, y_over = oversample.fit_resample(X, y)

# Train Logistic Regression model
lgr = LogisticRegression()
lgr.fit(X_over, y_over)

# Save the trained model
dump(lgr, 'log.pkl')
