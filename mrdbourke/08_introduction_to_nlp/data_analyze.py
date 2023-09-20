from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print("")
print("# Turn .csv files into pandas DataFrame's")
print(train_df.head())

print("")
print("# Shuffle training dataframe")
print(train_df_shuffled.head())

print("")
print("# The test data doesn't have a target (that's what we'd try to predict)")
print(test_df.head())

print("")
print("# How many examples of each class?")
print(train_df.target.value_counts())

print("")
print("# How many samples total?")
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")

print("")
random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")

print("")
print("# Check the lengths")
print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))

print("")
print("# View the first 10 training sentences and their labels")
print(train_sentences[:10], train_labels[:10])

print("")
print("# Find average number of tokens (words) in training Tweets")
print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))

print("")
print("# Create sample sentence and tokenize it")
sample_sentence = "There's a flood in my street!"
print(text_vectorizer([sample_sentence]))

print("")
print("# Choose a random sentence from the training dataset and tokenize it")
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nVectorized version:")
print(text_vectorizer([random_sentence]))

print("")
print("# Get the unique words in the vocabulary")
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-5:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}")
print(f"Bottom 5 least common words: {bottom_5_words}")

print("")
print(embedding)

print("")
print("# Get a random sentence from training set")
random_sentence = random.choice(train_sentences)
print(f"Original text:\n{random_sentence}\
      \n\nEmbedded version:")

print("# Embed the random sentence (turn it into numerical representation)")
sample_embed = embedding(text_vectorizer([random_sentence]))
print(sample_embed)

print("")
print("# Check out a single token's embedding")
print(sample_embed[0][0])

print("")
print("# Embedding of a sample sentence:")
sample_sentence = "Hiroshima: 70 years since the worst mass murder in human history. Never forget. http://t.co/jLu2J5QS8U"
print(sample_sentence)

print("")
sample_vectorized = text_vectorizer([sample_sentence])
print("# Sample vectorized:")
print(sample_vectorized)
   
print("")
print("# Sample embedded:")
sample_embed = embedding(sample_vectorized)
print(sample_embed)

print("")
print("# Embedding of the word `Hiroshima`")
print(sample_embed[0][0])

print("")
print("# Embedding of the word `the`")
print(sample_embed[0][4])

print("")
print("# Embedding of the word `in`")
print(sample_embed[0][8])
