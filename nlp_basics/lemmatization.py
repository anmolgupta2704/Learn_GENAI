import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = ["running", "better", "cars"]

lemmatized = [lemmatizer.lemmatize(word) for word in words]

print(lemmatized)