from model5_init import *

print()
print("# Get summary of token and character model")
model5.summary()

print()
print("# Plot hybrid token and character model")
plot_model(model5, to_file='data/images/plot_model5.png')
    
print()

