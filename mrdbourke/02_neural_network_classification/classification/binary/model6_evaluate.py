from model6_init import *

model6 = load_model("data/model6.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history6.pkl', 'rb') as f:
    history6 = pickle.load(f)

print("")
print("# Evaluate the model")
model6.evaluate(X, y)

print("")
print("# Check the deicison boundary (blue is blue class, yellow is the crossover, red is red class)")
plot_decision_boundary(model6, X, y, 6)
