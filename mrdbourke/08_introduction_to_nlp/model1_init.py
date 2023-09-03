from common import *

# Create tokenization and modelling pipeline
model1 = Pipeline( [
        ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
        ("clf", MultinomialNB()) # model the text
    ]
)

