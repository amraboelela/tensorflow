from common import *

model1 = load_model("data/model1.h5")

# Check the results of the insurance model
model1.evaluate(X_test_oh, y_test_oh)
