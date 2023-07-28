from model3_init import *

# Check to see if loaded model is a trained model
loaded_loss, loaded_accuracy = model.evaluate(test_data)
print(loaded_loss)
print(loaded_accuracy)

# Save the results to a file
with open('data/evaluation3.txt', 'w') as f:
    f.write(f'Test loss: {loaded_loss}\n')
    f.write(f'Test accuracy: {loaded_accuracy}\n')

# Make predictions with model
pred_probs = model.predict(test_data, verbose=1) # set verbosity to see how long it will take
print(len(pred_probs))
print(pred_probs.shape)
print(pred_probs[:10])

# We get one prediction probability per class
print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like:\n {pred_probs[0]}")
print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")

# Save the predictions to a file
np.savetxt('data/predictions3.txt', pred_probs)

