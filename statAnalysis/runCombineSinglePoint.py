#!/usr/bin/env python
import ROOT


import subprocess
import optparse
from Stat.Limits.settings import *

from Stat.Limits.combineUtils import runSinglePoint

 
usage = 'usage: %prog [--cat N]'
parser = optparse.OptionParser(usage)


parser.add_option("", "--mDark", dest='mDark',type="string", help='signal point mDark')
parser.add_option("", "--mZprime", dest='mZprime',type="string", help='signal point mZprime')
parser.add_option("", '--rinv', dest='rinv',type="string", help='signal point rinv')
parser.add_option("", '--alpha', dest='alpha', type="string",help='signal point alpha')
parser.add_option("-c", "--channel",dest="ch",type="string",default="all",help="Indicate channels of interest. Default is all")
parser.add_option("-y", "--years",dest="years",type="string",default="all",help="Indicate years of interest. Default is 2016")
parser.add_option('-m', '--method', dest='method', type='string', default = 'hist', help='Run a single method ( hist, template)')
parser.add_option('-d', '--dir', dest='dir', type='string', default = 'outdir', help='datacards direcotry')

parser.add_option('',"--runSingleCat",dest="runSingleCat",action='store_true', default=False)

(opt, args) = parser.parse_args()
#interaction = opt.interaction


path_ = "/t3home/decosa/SVJ/CMSSW_8_1_0/src/Stat/Limits/test/"
path_ += opt.dir

categories = channels
years = ["2016", "2017", "2018"]
if opt.years != "all":
    y_clean = opt.years.replace(" ", "")
    years = y_clean.split(",")



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

singleYear=""
if (len(years)==1 and opt.years!="all"):singleYear=years[0]
runSinglePoint(path_, opt.mZprime, opt.mDark, opt.rinv, opt.alpha, categories, opt.method, opt.runSingleCat, singleYear)
