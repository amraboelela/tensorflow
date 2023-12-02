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

