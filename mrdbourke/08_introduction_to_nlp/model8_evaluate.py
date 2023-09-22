from model1_evaluate import *
from model2_evaluate import *
from model3_evaluate import *
from model4_evaluate import *
from model5_evaluate import *
from model6_evaluate import *
from model7_evaluate import *
from model8_init import *

print("")
print("# Get a summary of the model")
print(model8.summary())

model8.load_weights(checkpoint_path(8))

print("")
print("# Make predictions on the validation dataset")
model8_pred_probs = model8.predict(val_sentences)
print(model8_pred_probs.shape, model8_pred_probs[:10], "# view the first 10")

print("")
print("# Round out predictions and reduce to 1-dimensional array")
model8_preds = tf.squeeze(tf.round(model8_pred_probs))
print(model8_preds[:10])

print("")
print("# Calculate model results")
model8_results = calculate_results(
    y_true=val_labels,
    y_pred=model8_preds
)
print(model8_results)

print("")
print("# Compare model8 to baseline")
compare_baseline_to_new_results(model1_results, model8_results)

print("")
print("# Combine model results into a DataFrame")
all_model_results = pd.DataFrame({
    "baseline": model1_results,
    "dense": model2_results,
    "lstm": model3_results,
    "gru": model4_results,
    "bidirectional": model5_results,
    "conv1d": model6_results,
    "use": model7_results,
    "10%": model8_results
})
all_model_results = all_model_results.transpose()
print(all_model_results)

print("")
print("# Reduce the accuracy to same scale as other metrics")
all_model_results["accuracy"] = all_model_results["accuracy"]/100

print("# Plot and compare all of the model results")
plt.figure()
all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));

plt.savefig('data/images/all_model_results.png', format='png')

print("")
print("# Sort model results by f1-score")
plt.figure()
all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar", figsize=(10, 7));
plt.savefig('data/images/all_model_f1_scores.png', format='png')

