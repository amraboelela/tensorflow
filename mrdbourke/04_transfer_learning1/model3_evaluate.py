from model3_init import *

model3.load_weights('data/model3.keras')

# Load the saved history object from a file
with open('data/history3.pkl', 'rb') as f:
    history3 = pickle.load(f)
    
print(model3.summary())
plot_curves(history3, 3)

subprocess.run(['tensorboard', '--logdir', './data/tensorflow_hub/'])
