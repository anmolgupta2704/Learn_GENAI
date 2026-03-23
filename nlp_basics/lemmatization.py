import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

words = ["running", "better", "cars"]

lemmatized = [lemmatizer.lemmatize(word) for word in words]

print(lemmatized)
tt = TweetTokenizer()
ss = SnowballStemmer("english")

text = "loving nlp !!! #ai"
tokens = tt.tokenize(text)

print(tokens)
print([ss.stem(t) for t in tokens])


text = "I love running"
tokens = tt.tokenize(text)

print(tokens)
print([ss.stem(t) for t in tokens])
text = "My studies are going on"
tokens = tt.tokenize(text)

print(tokens)
print([ss.stem(t) for t in tokens])