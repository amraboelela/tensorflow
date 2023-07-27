from model2_init import *

model2.load_weights('data/model2.keras')

# Load the saved history object from a file
with open('data/history2.pkl', 'rb') as f:
    history2 = pickle.load(f)

# Evaluate on the test data
model2.evaluate(test_data)

print(model2.summary())
plot_curves(history2, 2)
