import sys
sys.path.append("../training")

from module.DataLoader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle

qcd_path = "../../data/training_data/qcd/base_3/*.h5"
signal_path = "../../data/training_data/all_signals/1500GeV_0.15/base_3/*.h5"

model_output_path = "trainingResults/models/model_1500GeV_0.15.sav"

print("\n===================================")
print("Loading data")
print("===================================\n")


data_loader = DataLoader()
(X_train, Y_train), (X_test, Y_test) = data_loader.BDT_load_all_data(qcd_path=qcd_path, signal_path=signal_path)


print("\n===================================")
print("Loading model")
print("===================================\n")

# load the model from disk
bdt = pickle.load(open(model_output_path, 'rb'))


model_auc = roc_auc_score(Y_test, bdt.decision_function(X_test))
print("Area under ROC curve: %.4f"%(model_auc))

decisions = bdt.decision_function(X_test)
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(Y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()
