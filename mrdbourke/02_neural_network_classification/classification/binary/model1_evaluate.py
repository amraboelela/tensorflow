from model1_init import *

model1 = load_model("data/model1.keras")

print("")
print("# Train our model for longer (more chances to look at the data)")
model1.fit(X, y, epochs=200, verbose=0) # set verbose=0 to remove training updates
model1.evaluate(X, y)
