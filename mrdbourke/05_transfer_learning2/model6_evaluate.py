from model6_init import *

# Evaluate model (this is the fine-tuned 10 percent of data version)
model6_evaluate = read_tensor("model6_evaluate")
if model6_evaluate is None:
    model6_evaluate = model6.evaluate(test_data)
    save_tensor(model6_evaluate, "model6_evaluate")
print(model6_evaluate)

model6.load_weights(checkpoint_path(6))
  
model6_evaluate2 = read_tensor("model6_evaluate2")
if model6_evaluate2 is None:
    model6_evaluate2 = model6.evaluate(test_data)
    save_tensor(model6_evaluate2, "model6_evaluate2")
print(model6_evaluate2)

# Load the saved history object from a file
with open('data/history4.pkl', 'rb') as f:
    history4 = pickle.load(f)
    
# Load the saved history object from a file
with open('data/history6.pkl', 'rb') as f:
    history6 = pickle.load(f)
    
# How did fine-tuning go with more data?
compare_historys(
    original_history=history4,
    new_history=history6,
    initial_epochs=5
)
                 
plot_curves(history6, 6)
