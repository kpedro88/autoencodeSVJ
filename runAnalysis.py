import autoencodeSVJ.utils as utils
import autoencodeSVJ.trainer as trainer
import autoencodeSVJ.evaluate as ev

print("imports OK")

ev.update_all_signal_evals()

aucs = ev.load_auc_table()
s = utils.summary()