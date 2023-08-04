from model1_init import *

print(model1.summary())
exit()

model1.load_weights('data/model1.keras')

# Load the saved history object from a file
with open('data/history1.pkl', 'rb') as f:
    history1 = pickle.load(f)


plot_curves(history1, 1)

# Run in terminal % tensorboard --logdir ./data/transfer_learning
