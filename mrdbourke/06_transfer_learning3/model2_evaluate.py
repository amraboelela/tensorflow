from model1_init import *

print(model.summary())

compare_historys(original_history=history_all_classes_10_percent,
                 new_history=history_all_classes_10_percent_fine_tune,
                 initial_epochs=5)
subprocess.run(['mv', 'plot.png', imagePath + "/plot2.png"])

