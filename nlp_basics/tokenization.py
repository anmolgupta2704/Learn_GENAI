import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')

text = "GenAI is amazing. It is the future of AI."

print("Sentence Tokenization:")
print(sent_tokenize(text))

print("\nWord Tokenization:")
print(word_tokenize(text))