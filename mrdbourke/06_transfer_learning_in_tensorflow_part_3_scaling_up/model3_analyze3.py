from model3_evaluate_load import *

classification_report_dict = classification_report(y_labels, pred_classes, output_dict=True)
print(classification_report_dict)

