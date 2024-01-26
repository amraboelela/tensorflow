from model1_evaluate import *
from model2_evaluate import *
from model3_evaluate import *
from model4_evaluate import *
from model5_evaluate import *
from model6_init import *

print()
print("# Get a summary of our token, char and positional embedding model")
model6.summary()

model6.load_weights(checkpoint_path(6))

print()
print("# Plot the token, char, positional embedding model")
plot_model(model6, to_file='data/images/plot_model6.png')

# Check which layers of our model are trainable or not
for layer in model6.layers:
    print(layer, layer.trainable)

print()

print("# Make predictions with token-char-positional hybrid model")
model6_pred_probs = read_tensor("model6_pred_probs")
if model6_pred_probs is None:
    model6_pred_probs = model6.predict(val_pos_char_token_dataset, verbose=1)
    save_tensor(model6_pred_probs, "model6_pred_probs")
print(model6_pred_probs)

print()
print("# Turn prediction probabilities into prediction classes")
model6_preds = tf.argmax(model6_pred_probs, axis=1)
print(model6_preds)

print()
print("# Calculate results of token-char-positional hybrid model")
model6_results = calculate_results(
    y_true=val_labels_encoded,
    y_pred=model6_preds
)
print(model6_results)

print()
print("# Combine model results into a DataFrame")
all_model_results = pd.DataFrame({
    "baseline": model1_results,
    "custom_token_embed_conv1d": model2_results,
    "pretrained_token_embed": model3_results,
    "custom_char_embed_conv1d": model4_results,
    "hybrid_char_token_embed": model5_results,
    "tribrid_pos_char_token_embed": model6_results
})
all_model_results = all_model_results.transpose()
print(all_model_results)

print()
print("# Reduce the accuracy to same scale as other metrics")
all_model_results["accuracy"] = all_model_results["accuracy"]/100

print()
print("# Plot and compare all of the model results")
plt.figure()
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));
plt.savefig('data/images/all_model_results.png', format='png')

print()
print("# Sort model results by f1-score")
plt.figure()
all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7));
plt.savefig('data/images/all_model_f1_scores.png', format='png')

print()
print("# Create sentencizer - Source: https://spacy.io/usage/linguistic-features#sbd")
nlp = English() # setup English sentence parser

print()
print("# New version of spaCy")
sentencizer = nlp.add_pipe("sentencizer") # create sentence splitting pipeline object

print()
print("# Old version of spaCy")
print("# sentencizer = nlp.create_pipe('sentencizer') # create sentence splitting pipeline object")
print("# nlp.add_pipe(sentencizer) # add sentence splitting pipeline object to sentence parser")

print("# Create 'doc' of parsed sequences, change index for a different abstract")
doc = nlp(example_abstracts[0]["abstract"])
abstract_lines = [str(sent) for sent in list(doc.sents)] # return detected sentences from doc in string type (not spaCy token type)
print(abstract_lines)

print()
print("# Get total number of lines")
total_lines_in_sample = len(abstract_lines)

print()
print("# Go through each line in abstract and create a list of dictionaries containing features for each line")
sample_lines = []
for i, line in enumerate(abstract_lines):
    sample_dict = {}
    sample_dict["text"] = str(line)
    sample_dict["line_number"] = i
    sample_dict["total_lines"] = total_lines_in_sample - 1
    sample_lines.append(sample_dict)
print(sample_lines)

print()
print("# Get all line_number values from sample abstract")
test_abstract_line_numbers = [line["line_number"] for line in sample_lines]
print("# One-hot encode to same depth as training data, so model accepts right input shape")
test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
print(test_abstract_line_numbers_one_hot)

print()
print("# Get all total_lines values from sample abstract")
test_abstract_total_lines = [line["total_lines"] for line in sample_lines]
print("# One-hot encode to same depth as training data, so model accepts right input shape")
test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)
print(test_abstract_total_lines_one_hot)

print()
print("# Split abstract lines into characters")
abstract_chars = [split_chars(sentence) for sentence in abstract_lines]
print(abstract_chars)

print()
print("# Make predictions on sample abstract features")
test_abstract_pred_probs = model6.predict(
    x=(
        test_abstract_line_numbers_one_hot,
        test_abstract_total_lines_one_hot,
        tf.constant(abstract_lines),
        tf.constant(abstract_chars)
    )
)
print(test_abstract_pred_probs)

print()
print("# Turn prediction probabilities into prediction classes")
test_abstract_preds = tf.argmax(test_abstract_pred_probs, axis=1)
print(test_abstract_preds)

print()
print("# Turn prediction class integers into string class names")
test_abstract_pred_classes = [label_encoder.classes_[i] for i in test_abstract_preds]
print(test_abstract_pred_classes)

print()
print("# Visualize abstract lines and predicted sequence labels")
for i, line in enumerate(abstract_lines):
    print(f"{test_abstract_pred_classes[i]}: {line}")
  
print()

