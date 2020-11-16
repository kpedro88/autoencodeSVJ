import ROOT
import os
import sys
import optparse
from moduleStat.settings import *
from moduleStat.datacards import *

print("====> CHANNELS: ", channels)

path = "./"

usage = 'usage: %prog -p histosPath -o outputFile'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='ifile', type='string', default= path + "histos.root",help='Where can I find input histos? Default is histos.root')
parser.add_option("-d","--outdir",dest="outdir",type="string",default="outdir",help="Name of the output directory where to store datacards. Default is outdir")
parser.add_option("-m","--mode",dest="mode",type="string",default="hist",help="Kind of shape analysis: parametric fit or fit to histos?. Default is hist")
parser.add_option("-c","--channel",dest="ch",type="string",default="all",help="Indicate channels of interest. Default is all")
parser.add_option("-t", "--test", action="store_true", default=False, dest="bias")
parser.add_option("-u","--unblind",dest="unblind",action='store_true', default=False)

(opt, args) = parser.parse_args()
sys.argv.append('-b')

ifilename = opt.ifile

print("Filename: ", ifilename)

outdir = opt.outdir
mode = opt.mode
unblind = opt.unblind

bias = opt.bias

if opt.ch != "all": 
    ch_clean = opt.ch.replace(" ", "")
    channels = ch_clean.split(",")

signals = []

print("====> CHANNELS: ", channels)

for p in sigpoints:

    mZprime=p[0]
    mDark=p[1]
    rinv=p[2]
    alpha=p[3]

    print("Creating datacards for mZprime = ", mZprime, " GeV, mDark = ", mDark, " GeV, rinv = ", rinv, " , alpha = ", alpha)
    signal  = "SVJ_mZprime%s_mDark%s_rinv%s_alpha%s" % (mZprime, mDark, rinv, alpha) 
    #if(bias):     signal  = "SVJ_mZprime%s_mDark%s_rinv%s_alpha%s_bias" % (mZprime, mDark, rinv, alpha) 
    signals.append(signal)



#signals = ["SVJ_mZprime3000_mDark20_rinv03_alphapeak"]



    

try:
    ifile = ROOT.TFile.Open(ifilename)
except IOError:
    print("Cannot open ", ifilename)
else:
    print("Opening file ",  ifilename)
    ifile.cd()
    
    r = ROOT.gDirectory.GetListOfKeys()[0]
    
    r_years = [r.ReadObj().GetName()[-4:] for r in ROOT.gDirectory.GetListOfKeys() ]
    
    years =  list(set(r_years))
    

ch_year = []

print("====> CHANNELS: ", channels)
for y in years:
    channels_years = [ch + '_' + y for ch in channels ]
    ch_year= ch_year + channels_years
    

print("====> CHANNELS + YEAR: ", ch_year)

cmd = "rm ws.root"
os.system(cmd)

cmd = "rm Efficiencies.txt"
os.system(cmd)

effs = {}


ch_eff = ["SVJ0", "SVJ1", "SVJ2"]
ch_eff_year = []
for y in years:
    ch_eff_years = [ch + '_' + y for ch in ch_eff ]
    ch_eff_year= ch_eff_year + ch_eff_years


for s in signals:

    effs[s] = getEfficiency(s, ch_eff_year, ifilename)

y = json.dumps(effs)
outname =  "Efficiencies.txt"

efile = open(outname, 'w')
efile.write(y)
efile.close()



doModelling = True
for s in signals:

    if (signals.index(s)!=0): doModelling = False
    for ch in ch_year:
        getCard(s, ch, ifilename, outdir, doModelling, mode, bias)
