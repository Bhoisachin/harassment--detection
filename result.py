from joblib import load

import re, string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Load pre-trained model and vectorizer
log = load('log.pkl')
tfIdfVectorizer = load('tfidf_vectorizer.pkl')

# # Download required NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')

# Preprocessing function (same as training)
stop = stopwords.words('english')
regex = re.compile('[%s]' % re.escape(string.punctuation))

def preprocess_text(text):
    text = ' '.join([word for word in text.split() if word.lower() not in stop])
    text = regex.sub('', text)
    porter_stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    text = ' '.join([porter_stemmer.stem(word) for word in tokens if not word.isdigit()])
    return text

# Input text
sample = input("Enter the message: ").strip()
processed_sample = preprocess_text(sample)

# Transform using the loaded TfidfVectorizer
X_test = tfIdfVectorizer.transform([processed_sample]).toarray()

# Predict using the Logistic Regression model
y_pred = log.predict(X_test)
if y_pred[0]==1:
    print('this massage is cyberbulling')
    pass
else:
    print('this massage are not cyberbulling')
    pass

