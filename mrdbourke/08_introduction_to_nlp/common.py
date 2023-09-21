import sys
sys.path.append('../modules')
from helper_functions import *

# Download data (same as from Kaggle)
download("https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip")

# Turn .csv files into pandas DataFrame's
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
  
# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1, # dedicate 10% of samples to validation set
    random_state=42
) # random state for reproducibility

# Before TensorFlow 2.6
# Note: in TensorFlow 2.6+, you no longer need "layers.experimental.preprocessing"
# you can use: "tf.keras.layers.TextVectorization", see https://github.com/tensorflow/tensorflow/releases/tag/v2.6.0 for more

# Use the default TextVectorization variables
text_vectorizer = TextVectorization(
    max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
    standardize="lower_and_strip_punctuation", # how to process text
    split="whitespace", # how to split tokens
    ngrams=None, # create groups of n-words?
    output_mode="int", # how to map tokens to numbers
    output_sequence_length=None
) # how long should the output sequence of tokens be?
# pad_to_max_tokens=True) # Not valid if using max_tokens=None

# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(
    max_tokens=max_vocab_length,
    output_mode="int",
    output_sequence_length=max_length
)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)

embedding = Embedding(
    input_dim=max_vocab_length, # set input shape
    output_dim=128, # set size of embedding vector
    embeddings_initializer="uniform", # default, intialize randomly
    input_length=max_length, # how long is each input
    name="embedding_1"
)

inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = text_vectorizer(inputs) # turn the input text into numbers
x = embedding(x) # create an embedding of the numerized numbers
print(x.shape)
