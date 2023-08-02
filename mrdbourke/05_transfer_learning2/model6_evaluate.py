from model6_init import *

model6.load_weights(checkpoint_path(6))
    
# Load the saved history object from a file
with open('data/history6.pkl', 'rb') as f:
    history6 = pickle.load(f)
  
print(model6.summary())
plot_curves(history6, 6)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
                      
