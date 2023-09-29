from common import *

# Define a path to save the model
#model_path = "data/universal_sentence_encoder"
#if path.exists(model_path):
#    embed = tf.saved_model.load(model_path)
#else:
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") # load Universal Sentence Encoder
 #   tf.saved_model.save(embed, model_path)

embed_samples = embed([
    sample_sentence,
    "When you call the universal sentence encoder on a sentence, it turns it into numbers."
])

# We can use this encoding layer in place of our text_vectorizer and embedding layer
sentence_encoder_layer = KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder/4",
    input_shape=[], # shape of inputs coming to our model
    dtype=tf.string, # data type of inputs coming to the USE layer
    trainable=False, # keep the pretrained weights (we'll create a feature extractor)
    name="USE"
)

# Create model using the Sequential API
model7 = Sequential([
    sentence_encoder_layer, # take in sentences and then encode them into an embedding
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
], name="model7_USE")

# Compile model
model7.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

