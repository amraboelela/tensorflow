from common import *

# Load the evaluation results from the file
with open('data/evaluation3.txt', 'r') as f:
    result_lines = f.readlines()

# Parse the evaluation results
loaded_loss = float(result_lines[0].split(': ')[1])
loaded_accuracy = float(result_lines[1].split(': ')[1])

pred_probs = np.loadtxt('data/predictions3.txt')

