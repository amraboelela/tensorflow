from model1_init import *

print(model1.summary())

model1.load_weights(checkpoint_path(1))

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)

print("")
print("# Evaluate model")
results_feature_extraction_model = model1.evaluate(test_data)
print(results_feature_extraction_model)

plot_curves(history1, 1)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
