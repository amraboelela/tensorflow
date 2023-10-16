#from common2 import *
from model3_evaluate import *
from model4_evaluate import *

model5 = load_model("data/model5.keras")

model5.summary()

plot_model(model5, to_file='data/images/model5.png', show_shapes=True)

print("")
print("# Make and plot predictions for model4")
y_preds5 = model5.predict(X_test)
plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds5,
    index=5
)

print("")
print("# Calculate model5 metrics")
mae5 = mean_absolute_error(y_test, y_preds5.squeeze()).numpy()
mse5 = mean_squared_error(y_test, y_preds5.squeeze()).numpy()
print(mae5, mse5)

model_results = [["model3", mae3, mse3],
                 ["model4", mae4, mse4],
                 ["model5", mae5, mae5]]
                 
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results)

print("")
print("# Save a model using the SavedModel format")
model4.save('data/best_model_SavedModel_format')

print("# Save a model using the keras format")
model4.save("data/best_model_HDF5_format.keras") # note the addition of '.keras' on the end

print("# Load a model from the SavedModel format")
loaded_saved_model = load_model("data/best_model_SavedModel_format")
print(loaded_saved_model.summary())

print("")
print("# Compare model4 with the SavedModel version (should return True)")
model4_preds = model4.predict(X_test)
saved_model_preds = loaded_saved_model.predict(X_test)
print(mean_absolute_error(y_test, saved_model_preds.squeeze()).numpy() == mean_absolute_error(y_test, model4_preds.squeeze()).numpy())

print("")
print("# Load a model from the HDF5 format")
loaded_h5_model = load_model("data/best_model_HDF5_format.keras")
print(loaded_h5_model.summary())

print("")
print("# Compare model4 with the loaded HDF5 version (should return True)")
h5_model_preds = loaded_h5_model.predict(X_test)
print(mean_absolute_error(y_test, h5_model_preds.squeeze()).numpy() == mean_absolute_error(y_test, model4_preds.squeeze()).numpy())

