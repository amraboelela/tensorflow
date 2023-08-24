from model1_init import *

print(model1.summary())

model1.load_weights(checkpoint_path(1))

print("")
print("# Evaluate model")
model1_evaluate = read_tensor("model1_evaluate")
if model1_evaluate is None:
    model1_evaluate = model1.evaluate(test_data)
    save_tensor(model1_evaluate, "model1_evaluate")
print(model1_evaluate)

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
    
plot_curves(history1, 1)
