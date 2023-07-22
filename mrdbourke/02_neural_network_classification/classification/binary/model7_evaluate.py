from model7_init import *

model7 = load_model("data/model7.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history7.pkl', 'rb') as f:
    history7 = pickle.load(f)

print("")
print("# Evaluate the model")
model7.evaluate(X, y)

print("")
print("# Check the deicison boundary (blue is blue class, yellow is the crossover, red is red class)")
plot_decision_boundary(model7, X, y, 7)
