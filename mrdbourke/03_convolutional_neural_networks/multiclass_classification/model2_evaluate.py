from model2_init import *

model2.load_weights('data/model2.keras')

# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

# Evaluate on the test data
model2_evaluate = read_tensor("model2_evaluate")
if model2_evaluate is None:
    model2_evaluate = model2.evaluate(test_data)
    save_tensor(model2_evaluate, "model2_evaluate")
print(model2_evaluate)

print(model2.summary())
plot_curves(history2, 2)
