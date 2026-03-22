from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "runner", "runs", "ran"]

stemmed_words = [stemmer.stem(word) for word in words]

print(stemmed_words)