import sys
sys.path.append("../training")


from module.DataLoader import DataLoader
import module.utils as utils

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

qcd_path = "../../data/training_data/qcd/base_3/*.h5"
signal_path = "../../data/training_data/all_signals/1500GeV_0.15/base_3/*.h5"

print("hello")



def BDT_load_all_data(test_split=0.2,
                      random_state=-1,
                      include_hlf=True, include_eflow=True,
                      hlf_to_drop=['Energy', 'Flavor']):
    """General-purpose data loader for BDT training, which separates classes and splits data into training/testing data.

    Args:
        SVJ_path (str): glob-style specification of .h5 files to load as SVJ signal
        qcd_path (str): glob-style specification of .h5 files to load as qcd background
        test_split (float): fraction of total data to use for testing
        random_state (int): random seed, leave as -1 for random assignment
        include_hlf (bool): true to include high-level features in loaded data, false for not
        include_eflow (bool): true to include energy-flow basis features in loaded data, false for not
        hlf_to_drop (list(str)): list of high-level features to drop from the final dataset. Defaults to dropping Energy and Flavor.

    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame): X,Y training data, where X is the data samples for each jet, and Y is the
            signal/background tag for each jet
        tuple(pandas.DataFrame, pandas.DataFrame): X_test,Y_test testing data, where X are data samples for each jet and Y is the
            signal/background tag for each jet
    """
    
    if random_state < 0:
        random_state = np.random.randint(0, 2 ** 32 - 1)

    data_loader = DataLoader()

    # Load QCD samples
    
    (QCD, _, _, _) = data_loader.load_all_data(qcd_path, "QCD",
                                               include_hlf=include_hlf, include_eflow=include_eflow,
                                               hlf_to_drop=hlf_to_drop)

    (SVJ, _, _, _) = data_loader.load_all_data(signal_path, "SVJ",
                                               include_hlf=include_hlf, include_eflow=include_eflow,
                                               hlf_to_drop=hlf_to_drop)

    SVJ_train, SVJ_test = train_test_split(SVJ.df, test_size=test_split, random_state=random_state)
    QCD_train, QCD_test = train_test_split(QCD.df, test_size=test_split, random_state=random_state)
    
    SVJ_Y_train, SVJ_Y_test = [pd.DataFrame(np.ones((len(elt), 1)), index=elt.index, columns=['tag']) for elt in [SVJ_train, SVJ_test]]
    QCD_Y_train, QCD_Y_test = [pd.DataFrame(np.zeros((len(elt), 1)), index=elt.index, columns=['tag']) for elt in [QCD_train, QCD_test]]
    
    X = SVJ_train.append(QCD_train)
    Y = SVJ_Y_train.append(QCD_Y_train)
    
    X_test = SVJ_test.append(QCD_test)
    Y_test = SVJ_Y_test.append(QCD_Y_test)
    
    return (X, Y), (X_test, Y_test)


train, test = BDT_load_all_data()

X_train = train[0]
Y_train = train[1]

X_test = test[0]
Y_test = test[1]

print("\n===================================")
print("Fitting a model")
print("===================================\n")
bdt = AdaBoostClassifier(algorithm='SAMME', n_estimators=800, learning_rate=0.5)
bdt.fit(X_train, Y_train)

print("\n===================================")
print("Fit done")
print("===================================\n")

Y_predicted = bdt.predict(X_test)


print(classification_report(Y_test, Y_predicted, target_names=["background", "signal"]))

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