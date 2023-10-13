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
