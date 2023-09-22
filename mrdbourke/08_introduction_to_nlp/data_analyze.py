from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print(x.shape)

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

print("")
print("# Test out the embedding, 1D convolutional and max pooling")
embedding_test = embedding(text_vectorizer(["this is a test sentence"])) # turn target sentence into embedding
conv_1d = Conv1D(filters=32, kernel_size=5, activation="relu") # convolve over target sequence 5 words at a time
conv_1d_output = conv_1d(embedding_test) # pass embedding through 1D convolutional layer
max_pool = GlobalMaxPool1D()
max_pool_output = max_pool(conv_1d_output) # get the most important features
print(embedding_test.shape, conv_1d_output.shape, max_pool_output.shape)

print("")
print("# See the outputs of each layer")
print(embedding_test[:1], conv_1d_output[:1], max_pool_output[:1])

print("")
print("# Example of pretrained embedding with universal sentence encoder - https://tfhub.dev/google/universal-sentence-encoder/4")
print(embed_samples[0][:50])

print("")
print("# Each sentence has been encoded into a 512 dimension vector")
print(embed_samples[0].shape)

print("")
print("# Check length of 10 percent datasets")
print(f"Total training examples: {len(train_sentences)}")
print(f"Length of 10% training examples: {len(train_sentences_10_percent)}")

print("")
print("# Check the number of targets in our subset of data")
print("# (this should be close to the distribution of labels in the original train_labels)")
print(pd.Series(train_labels_10_percent).value_counts())

