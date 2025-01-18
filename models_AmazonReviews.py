import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv("./dataset/AmazonReviews/train.csv", names=["polarity", "title", "text"])
print('reading done.')

# Download the NLTK resources
nltk.download('stopwords')
nltr=nltk.download('wordnet')

# Preprocess the text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)         # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)                 # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)          # Remove digits
    text = text.lower()                                 # Convert to lowercase
    words = text.split()                                # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words] # Remove stopwords and lemmatize
    return ' '.join(words)                             # Join the words back into a string

# Apply the preprocessing to the text column
data["text"] = data["text"].apply(preprocess_text)
print('preprocessing done.')

# Split the data into training and testing sets
X = data["text"]
y = data["polarity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('splitting done.')

# Convert the text into TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print('vectorizing done.')

# ML Models - Logistic Regression, SVM, Random Forest and so on...
# Train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

#Deep Learning Models - LSTM, BERT, RNN and so on...
# Train an LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_tfidf, y_train, epochs=5, batch_size=64, validation_data=(X_test_tfidf, y_test))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()