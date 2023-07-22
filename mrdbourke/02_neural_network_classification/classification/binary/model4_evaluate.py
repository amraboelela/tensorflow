from model4_init import *

model4 = load_model("data/model4.keras")

print("")
print("# Load the saved history object from a file")
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)

print("")
print("# Check out our data")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.savefig('data/images/xy.png', format='png')

print("")
print("# Check the deicison boundary (blue is blue class, yellow is the crossover, red is red class)")
plot_decision_boundary(model4, X, y, 4)
