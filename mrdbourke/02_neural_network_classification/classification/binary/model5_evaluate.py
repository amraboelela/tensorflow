from model5_init import *

model5 = load_model("data/model5.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history5.pkl', 'rb') as f:
    history5 = pickle.load(f)

print("")
print("# Check the deicison boundary (blue is blue class, yellow is the crossover, red is red class)")
plot_decision_boundary(model5, X, y, 5)
