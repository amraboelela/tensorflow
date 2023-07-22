from common1 import *

model1 = load_model("data/model1.keras")

print("")
print("# Check out X and y")
print(X, y)

print("")
print("# Make a prediction with the model")
print(model1.predict([17.0]))
