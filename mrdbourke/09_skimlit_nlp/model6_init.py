from common import *

# 1. Token inputs
token_inputs = layers.Input(shape=[], dtype="string", name="token_inputs")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_outputs = Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(
    inputs=token_inputs,
    outputs=token_outputs
)

# 2. Char inputs
char_inputs = layers.Input(shape=(1,), dtype="string", name="char_inputs")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = Bidirectional(layers.LSTM(32))(char_embeddings)
char_model = tf.keras.Model(
    inputs=char_inputs,
    outputs=char_bi_lstm
)

# 3. Line numbers inputs
line_number_inputs = layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
x = Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.Model(
    inputs=line_number_inputs,
    outputs=x
)

# 4. Total lines inputs
total_lines_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
y = Dense(32, activation="relu")(total_lines_inputs)
total_line_model = tf.keras.Model(
    inputs=total_lines_inputs,
    outputs=y
)

# 5. Combine token and char embeddings into a hybrid embedding
combined_embeddings = Concatenate(name="token_char_hybrid_embedding")([
    token_model.output,
    char_model.output
])
z = Dense(256, activation="relu")(combined_embeddings)
z = Dropout(0.5)(z)

# 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
z = Concatenate(name="token_char_positional_embedding")([
    line_number_model.output,
    total_line_model.output,
    z
])

# 7. Create output layer
output_layer = layers.Dense(5, activation="softmax", name="output_layer")(z)

# 8. Put together model
model6 = tf.keras.Model(inputs=[
        line_number_model.input,
        total_line_model.input,
        token_model.input,
        char_model.input
    ],
    outputs=output_layer
)
