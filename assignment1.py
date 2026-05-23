import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

feedback = [
    'the faculty was very supportive and the lectures were really helpful',
    'i am not satisfied with the lab sessions',
    'the courser content is good',
    'lan were not working properly'
]

ss = SnowballStemmer("english")

tokens_list = []

# Tokenization
for sentence in feedback:
    tokens = word_tokenize(sentence)
    tokens_list.append(tokens)

print("Tokens:", tokens_list)

# Stopword removal
stop_word = set(stopwords.words('english'))
filtered_tokens = []

for tokens in tokens_list:
    filtered = [word for word in tokens if word.lower() not in stop_word]
    filtered_tokens.append(filtered)

print("After Stopword Removal:", filtered_tokens)

stemmed_tokens = []
for tokens in filtered_tokens:
    stemmed = [ss.stem(word) for word in tokens]
    stemmed_tokens.append(stemmed)

print("After Stemming:", stemmed_tokens)

from functions.otp_generation import generate_alphanumeric_otp
print("Generated OTP:", generate_alphanumeric_otp())
