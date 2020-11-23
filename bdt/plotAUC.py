import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



iname = "ROCauc.txt"
#iname = "RocCurves_0p45.txt"
#ifile = open(iname, 'r')
#iline = ifile.readline()
#aucs_ = json.loads(iline)

aucs = {}
with open(iname) as json_file:
    aucs_ = json.load(json_file)

#print "AUCs ", aucs_
#mzprime = ["1500", "2000"]
#rinv = ["0.15", "0.30"]
mzprime = ["1500", "2000", "2500", "3000", "3500", "4000"]
#mzprime = ["1500", "2000"]
rinv = ["0.15", "0.30", "0.45", "0.60", "0.75"]
#rinv = [ "0.15","0.45"]

#rinv = ["0.15", "0.30", "0.60", "0.75"]



auc_values = []
for m in mzprime:
    rinvs = []
    for r in rinv:
        sig = "%sGeV_%s"%(m, r)
        rinvs.append(int(aucs_[sig]*1000)/1000.)
    auc_values.append(rinvs)

aucs = np.array(auc_values)

print "AUCs ", aucs

fig, ax = plt.subplots()
im = ax.imshow(aucs)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("AUC", rotation=-90, va="bottom")


# We want to show all ticks...
ax.set_xticks(np.arange(len(rinv)))
ax.set_yticks(np.arange(len(mzprime)))
# ... and label them with the respective list entries
ax.set_xticklabels(rinv)
ax.set_yticklabels(mzprime)

ax.set_xlabel("$r_{inv}$")
ax.set_ylabel("$M_{Z^\prime}$")

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")



for r in xrange(len(rinv)):
    for m in xrange(len(mzprime)):

        sig = "%sGeV_%s"%(mzprime[m],rinv[r])
        print "Signal ", sig
#        text = ax.text(m, r, aucs_[sig],
        text = ax.text(r, m, aucs[m,r],
                       ha="center", va="center", color="w")

ax.set_title("Classifier performance")
fig.tight_layout()

plt.savefig("plots/bdtAUCs_.png")
