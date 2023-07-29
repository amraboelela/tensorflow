from model1_init import *

print(model.summary())
plot_curves(history_all_classes_10_percent, 1)
subprocess.run(['mv', 'plot.png', imagePath + "/plot1.png"])

