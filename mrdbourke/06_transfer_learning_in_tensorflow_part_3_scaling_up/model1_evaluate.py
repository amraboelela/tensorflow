from model1_load import *

print(model.summary())
plot_loss_curves(history_all_classes_10_percent)
subprocess.run(['mv', 'plot.png', imagePath + "/plot1.png"])
