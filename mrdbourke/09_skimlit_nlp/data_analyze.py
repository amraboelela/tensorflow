from common import *

print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

print(filenames)

print()
print("# the whole first example of an abstract + a little more of the next one")
print(train_lines[:20])

print(len(train_samples), len(val_samples), len(test_samples))

print()
print("# Check the first abstract of our training data")
print(train_samples[:14])

print()
print(train_df.head(14))

print()
print("# Distribution of labels in training data")
print(train_df.target.value_counts())

train_df.total_lines.plot.hist()
plt.savefig('data/images/train_df.png', format='png')

print()
print("# Convert abstract text lines into lists")
print(len(train_sentences), len(val_sentences), len(test_sentences))

print()
print("# View first 10 lines of training sentences")
print(train_sentences[:10])

print()
print("# Check what training labels look like")
print(train_labels_one_hot)

print()
print("# Check what training labels look like")
print(train_labels_encoded)

print()
print(num_classes, class_names)

print()
print("# How long is each sentence on average?")
sent_lens = [len(sentence.split()) for sentence in train_sentences]
avg_sent_len = np.mean(sent_lens)
print(avg_sent_len, "# return average sentence length (in tokens)")

print()
print("# What's the distribution look like?")
plt.figure()
plt.hist(sent_lens, bins=7)
plt.savefig('data/images/sent_lens.png', format='png')

print()
print("# How long of a sentence covers 95% of the lengths?")
output_seq_len = int(np.percentile(sent_lens, 95))
print(output_seq_len)

print()
print("# Maximum sentence length in the training set")
print(max(sent_lens))

print()
print("# Test out text vectorizer")
target_sentence = random.choice(train_sentences)
print(f"Text:\n{target_sentence}")
print(f"\nLength of text: {len(target_sentence.split())}")
print(f"\nVectorized text:\n{text_vectorizer([target_sentence])}")

print()
print(f"Number of words in vocabulary: {len(rct_20k_text_vocab)}"),
print(f"Most common words in the vocabulary: {rct_20k_text_vocab[:5]}")
print(f"Least common words in the vocabulary: {rct_20k_text_vocab[-5:]}")

print()
print("# Get the config of our text vectorizer")
print(text_vectorizer.get_config())

print()
print("# Show example embedding")
print(f"Sentence before vectorization:\n{target_sentence}\n")
vectorized_sentence = text_vectorizer([target_sentence])
print(f"Sentence after vectorization (before embedding):\n{vectorized_sentence}\n")
embedded_sentence = token_embed(vectorized_sentence)
print(f"Sentence after embedding:\n{embedded_sentence}\n")
print(f"Embedded sentence shape: {embedded_sentence.shape}")

print(train_dataset)

print()
print("# Test out the embedding on a random sentence")
print(f"Random training sentence:\n{random_training_sentence}\n")
use_embedded_sentence = tf_hub_embedding_layer([random_training_sentence])
print(f"Sentence after embedding:\n{use_embedded_sentence[0][:30]} (truncated output)...\n")
print(f"Length of sentence embedding:\n{len(use_embedded_sentence[0])}")

print()
print("# Test splitting non-character-level sequence into characters")
print(split_chars(random_training_sentence))

print()
print("# Split sequence-level data splits into character-level data splits")
print(train_chars[0])

print()
print("# What's the average character length?")
print(mean_char_len)

print()
print("# Check the distribution of our sequences at character-level")
plt.figure()
plt.hist(char_lens, bins=7)
plt.savefig('data/images/char_lens.png', format='png')

print()
print("# Find what character length covers 95% of sequences")
print(output_seq_char_len)

print()
print("# Get all keyboard characters for char-level embedding")
print(alphabet)

print()
print("# Check character vocabulary characteristics")
char_vocab = char_vectorizer.get_vocabulary()
print(f"Number of different characters in character vocab: {len(char_vocab)}")
print(f"5 most common characters: {char_vocab[:5]}")
print(f"5 least common characters: {char_vocab[-5:]}")

print()
print("# Test out character vectorizer")
random_train_chars = random.choice(train_chars)
print(f"Charified text:\n{random_train_chars}")
print(f"\nLength of chars: {len(random_train_chars.split())}")
vectorized_chars = char_vectorizer([random_train_chars])
print(f"\nVectorized chars:\n{vectorized_chars}")
print(f"\nLength of vectorized chars: {len(vectorized_chars[0])}")

print()
print("# Test out character embedding layer")
print(f"Charified text (before vectorization and embedding):\n{random_train_chars}\n")
char_embed_example = char_embed(char_vectorizer([random_train_chars]))
print(f"Embedded chars (after vectorization and embedding):\n{char_embed_example}\n")
print(f"Character embedding shape: {char_embed_example.shape}")

print()
print("# Create char datasets")
print(train_char_dataset)

print()
print("# Check out training char and token embedding dataset")
print(train_char_token_dataset, val_char_token_dataset)

print()
print("# Inspect training dataframe")
print(train_df.head())

print()
print("# How many different line numbers are there?")
print(train_df["line_number"].value_counts())

print()
print("# Check the distribution of 'line_number' column")
plt.figure()
train_df.line_number.plot.hist()
plt.savefig('data/images/train_df_line_number.png', format='png')

print()
print("# Check one-hot encoded 'line_number' feature samples")
print(train_line_numbers_one_hot.shape, train_line_numbers_one_hot[:20])

print()
print("# How many different numbers of lines are there?")
print(train_df["total_lines"].value_counts())

print()
print("# Check the distribution of total lines")
plt.figure()
train_df.total_lines.plot.hist();
plt.savefig('data/images/train_df_total_lines.png', format='png')

print()
print("# Check the coverage of a 'total_lines' value of 20")
print(np.percentile(train_df.total_lines, 98)) # a value of 20 covers 98% of samples

print()
print("# Check shape and samples of total lines one-hot tensor")
print(train_total_lines_one_hot.shape, train_total_lines_one_hot[:10])

print()
print("# Check input shapes")
print(train_pos_char_token_dataset, val_pos_char_token_dataset)

print()
print("# Download and open example abstracts (copy and pasted from PubMed)")
print(example_abstracts)

print()
print("# See what our example abstracts look like")
abstracts = pd.DataFrame(example_abstracts)
print(abstracts)

print()
