from model1_evaluate import *
from model2_init import *

print("")
print("# Get a summary of the model")
print(model2.summary())

model2.load_weights(checkpoint_path(2))

print("")
print("# Evaluate model")
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(val_sentences, val_labels)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

print(embedding.weights)

embed_weights = model2.get_layer("embedding_1").get_weights()[0]
print(embed_weights.shape)

print("")
print("# Make predictions (these come back in the form of probabilities)")
model2_pred_probs = model2.predict(val_sentences)
print(model2_pred_probs[:10], "# only print out the first 10 prediction probabilities")

print("")
print("# Turn prediction probabilities into single-dimension tensor of floats")
model2_preds = tf.squeeze(tf.round(model2_pred_probs)) # squeeze removes single dimensions
print(model2_preds[:20])

print("")
print("# Calculate model2 metrics")
model2_results = calculate_results(
    y_true=val_labels,
    y_pred=model2_preds
)
print(model2_results)

print("")
print("# Is our simple Keras model better than our baseline model?")
print(np.array(list(model2_results.values())) > np.array(list(baseline_results.values())))

compare_baseline_to_new_results(
    baseline_results=baseline_results,
    new_model_results=model2_results
)

print("")
print("# Get the vocabulary from the text vectorization layer")
words_in_vocab = text_vectorizer.get_vocabulary()
print(len(words_in_vocab), words_in_vocab[:10])

# Code below is adapted from: https://www.tensorflow.org/tutorials/text/word_embeddings#retrieve_the_trained_word_embeddings_and_save_them_to_disk

print("")
print("# Create output writers")
out_v = io.open("data/embedding_vectors.tsv", "w", encoding="utf-8")
out_m = io.open("data/embedding_metadata.tsv", "w", encoding="utf-8")

print("")
print("# Write embedding vectors and words to file")
for num, word in enumerate(words_in_vocab):
    if num == 0:
        continue # skip padding token
    vec = embed_weights[num]
    out_m.write(word + "\n") # write words to file
    out_v.write("\t".join([str(x) for x in vec]) + "\n") # write corresponding word vector to file
out_v.close()
out_m.close()

print("")
print("# Download files locally to upload to Embedding Projector")
try:
    from google.colab import files
except ImportError:
    pass
else:
    files.download("data/embedding_vectors.tsv")
    files.download("data/embedding_metadata.tsv")
