from model1_init import *

model1.load_weights('data/model1.keras')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)

# Evaluate on the test data
model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(test_data)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)

print(model1.summary())
plot_curves(history1, 1)
