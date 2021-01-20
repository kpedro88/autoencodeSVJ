import sys

sys.path.append("../training")

from module.DataLoader import DataLoader
from sklearn.metrics import roc_auc_score
import pickle

qcd_path = "../../data/training_data/qcd/base_3/*.h5"
out_file_name = "trainingResults/aucs.txt"

def get_auc_for_sample(mass, rinv):
    signal_path = "../../data/training_data/all_signals/{}GeV_{:3.2f}/base_3/*.h5".format(mass, rinv)
    model_output_path = "trainingResults/models/model_{}GeV_{:3.2f}.sav".format(mass, rinv)
    
    data_loader = DataLoader()
    (_, _), (X_test, Y_test) = data_loader.BDT_load_all_data(qcd_path=qcd_path, signal_path=signal_path)
    
    bdt = pickle.load(open(model_output_path, 'rb'))
    model_auc = roc_auc_score(Y_test, bdt.decision_function(X_test))
    return model_auc

with open(out_file_name, "w") as out_file:
    
    for mass in [1500, 2000, 2500, 3000, 3500, 4000]:
        for rinv in [0.15, 0.30, 0.45, 0.60, 0.75]:
            
            model_auc = get_auc_for_sample(mass, rinv)
            print("Area under ROC curve: %.4f" % (model_auc))
            out_file.write("{}\t{}\t{}\n".format(mass, rinv, model_auc))
            
            
    
    

# decisions = bdt.decision_function(X_test)
# # Compute ROC curve and area under the curve
# fpr, tpr, thresholds = roc_curve(Y_test, decisions)
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.grid()
# plt.show()
