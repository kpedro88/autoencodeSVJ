import os
import subprocess
import optparse
from Stat.Limits.settings import *
from Stat.Limits.combineUtils import *

usage = 'usage: %prog [--cat N]'
parser = optparse.OptionParser(usage)
#parser.add_option('', '--mzprime', dest='mzprime', type='int', default = 1, help='Mediator mass')
#parser.add_option('', '--mdark', dest='mdark', type='int', default = 1, help='Dark mass')
parser.add_option("-c","--channel",dest="ch",type="string",default="all",help="Indicate channels of interest. Default is all")
parser.add_option("-y","--years",dest="years",type="string",default="all",help="Indicate years of interest. Default is 2016")
parser.add_option('-s', '--syst', dest='syst', type='string', default = '1', help='Set the flag to 0 to remove systematics')
parser.add_option('-m', '--method', dest='method', type='string', default = 'hist', help='Run a single method (all, hist, template)')
parser.add_option('-S', '--sig', dest='sig', type='int', default = 0, help='Set the flag to 1 to enable significance computation')
parser.add_option('-d', '--dir', dest='dir', type='string', default = 'outdir', help='datacards direcotry')
parser.add_option('',"--runSingleCat",dest="runSingleCat",action='store_true', default=False)

(opt, args) = parser.parse_args()
#interaction = opt.interaction


path_ = "/t3home/decosa/SVJ/CMSSW_8_1_0/src/Stat/Limits/test/"
path_ += opt.dir



years = ["2016", "2017", "2018"]
if opt.years != "all": 
    y_clean = opt.years.replace(" ", "")
    years = y_clean.split(",")

categories = [ "lowSVJ1", "lowSVJ2", "highSVJ1", "highSVJ2"]
methods = ["hist", "template"]
if opt.ch != "all": 
    ch_clean = opt.ch.replace(" ", "")
    categories = ch_clean.split(",")

if opt.method != "all": 
    meth_clean = opt.method.replace(" ", "")
    methods = meth_clean.split(",")
    

cats = []
for y in years:
    
    cats_ = [c + "_" + y for c in categories]
    cats = cats + cats_

categories = cats


print "Combinining the following categories: ", categories



for point in sigpoints:

    mZprime=point[0]
    mDark=point[1]
    rinv=point[2]
    alpha=point[3]

    for method in methods:
        runSinglePoint(path_, mZprime, mDark, rinv, alpha, categories, method, opt.runSingleCat)

