from model3_load import *

# Check to see if loaded model is a trained model
loaded_loss, loaded_accuracy = model.evaluate(test_data)
print(loaded_loss)
print(loaded_accuracy)

