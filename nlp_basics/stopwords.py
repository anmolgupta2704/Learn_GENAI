import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

text = "This is a simple example of stopwords removal."

words = word_tokenize(text)

filtered = [word for word in words if word.lower() not in stopwords.words('english')]

print(filtered)