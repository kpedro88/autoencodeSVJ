import sys
sys.path.append("../training")

from module.DataLoader import DataLoader

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

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
print("Fitting a model")
print("===================================\n")
bdt = AdaBoostClassifier(algorithm='SAMME', n_estimators=800, learning_rate=0.5)
bdt.fit(X_train, Y_train)

print("\n===================================")
print("Fit done. Saving model")
print("===================================\n")

pickle.dump(bdt, open(model_output_path, 'wb'))

Y_predicted = bdt.predict(X_test)
report = classification_report(Y_test, Y_predicted, target_names=["background", "signal"])
print("Report: ", report)