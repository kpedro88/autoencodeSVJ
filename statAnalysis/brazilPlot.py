
import ROOT
import collections
import optparse
#ROOT.gROOT.Reset();
#ROOT.gROOT.SetStyle('Plain')
#ROOT.gStyle.SetPalette(1)
#ROOT.gStyle.SetOptStat(0)
#ROOT.gROOT.SetBatch()        # don't pop up canvases
#ROOT.TH1.AddDirectory(False)
from array import array
from samples.toPlot import samples
from Stat.Limits.settings import *





class limit(object):
    pass

def readFile(filename, cat):

    v = []
    o = []
    u1 = []
    u2 = []
    d1 = []
    d2 = []
    
    ifile = open(filename)

    print "Reading ", filename
    for l in ifile.readlines():


        if l.strip()== "": continue
        else:

            l_split = l.split() 
            if l.startswith("y_vals_"+cat):  v = [float(i) for i in l_split[1:]  ]
            elif l.startswith("y_observed_"+cat): o = [float(i) for i in l_split[1:] ]
            elif l.startswith("y_up_points1_"+cat): u1 = [float(i) for i in l_split[1:] ]
            elif l.startswith("y_up_points2_"+cat): u2 = [float(i) for i in l_split[1:] ]
            elif l.startswith("y_down_points1_"+cat): d1 = [float(i) for i in l_split[1:] ]
            elif l.startswith("y_down_points2_"+cat): d2 = [float(i) for i in l_split[1:] ]

#            if l.startswith("y_vals_"+cat):  v = [float(i) for i in l_split[1:] if float(i)!= 0.0 ]
#            elif l.startswith("y_observed_"+cat): o = [float(i) for i in l_split[1:] if float(i)!= 0.0]
#            elif l.startswith("y_up_points1_"+cat): u1 = [float(i) for i in l_split[1:] if float(i)!= 0.0]
#            elif l.startswith("y_up_points2_"+cat): u2 = [float(i) for i in l_split[1:] if float(i)!= 0.0]
#            elif l.startswith("y_down_points1_"+cat): d1 = [float(i) for i in l_split[1:] if float(i)!= 0]
#            elif l.startswith("y_down_points2_"+cat): d2 = [float(i) for i in l_split[1:] if float(i)!= 0.0]
            #print l      

    limits = limit()
    limits.v = v
    limits.o = o
    limits.u1 = u1
    limits.u2 = u2
    limits.d1 = d1
    limits.d2 = d2
    
    return limits



#def plotLimits(cat, year, method):

usage = 'usage: %prog --method method'
parser = optparse.OptionParser(usage)
parser.add_option("-r","--ratio",dest="ratio",action='store_true', default=False)
parser.add_option('-m', '--method', dest='method', type='string', default = 'hist', help='Run a single method (all, hist, template)')
parser.add_option('-y', '--year', dest='year', type='string', default = 'all', help='Run a single method (Run2, 2016, 2017, 2016_2017)')
parser.add_option('-v', '--variable', dest='variable', type='string', default = 'mZprime', help='Plot limit against variable v (mZPrime, mDark, rinv, alpha)')
parser.add_option("-u","--unblind",dest="unblind",action='store_true', default=False)
parser.add_option("-c","--cat",dest="cat",type="string",default="",help="Indicate channels of interest. Default is all")
(opt, args) = parser.parse_args()

unblind = opt.unblind
theo = not opt.ratio


if(opt.cat!=""):
    filename = "data/limit_%s_%s.txt" % (opt.cat, opt.method)
    cat ="%s_%s"% (opt.cat, opt.method)
else:
    filename = "data/limit_%s.txt" % (opt.method)
    cat = opt.method    


l = readFile(filename, cat)



xvalues_ = []
for point in sigpoints:


    var = point[0]    
    if opt.variable == "mDark": var = point[1]
    elif opt.variable == "rinv": var = point[2]
    elif opt.variable == "alpha": var = point[3]
    
    xvalues_.append(int(var))



print "Mass points: ", xvalues_
print "Min: ", min(xvalues_)
print "Max: ", max(xvalues_)
print "l.v ", l.v

#print "l.u1 ", l.u1

print "Reading file: ", ("data/limit_%s.txt" % (opt.method))
ebar_u1 = [l.u1[i] - l.v[i] for i in xrange(len(l.v))]
ebar_u2 = [l.u2[i] - l.v[i] for i in xrange(len(l.v))]
ebar_d1 = [l.v[i] - l.d1[i] for i in xrange(len(l.v))]
ebar_d2 = [l.v[i] - l.d2[i] for i in xrange(len(l.v))]

#print "u2 ", l.u2
#print "u1 ", l.u1
#print "median ", l.v
#print "d1 ", l.d1
#print "d2 ", l.d2

#print "U2 ", ebar_u2
#print "U1 ", ebar_u1


med_values = array('f', l.v)

obs_values = array('f', l.o)

xvalues = array('f', xvalues_)

print "XVALUES: ", xvalues

print "MEDVALUES: ", med_values

y_theo = {str(mass):samples["SVJ_mZprime%d_mDark20_rinv03_alphapeak" % (mass)].sigma for mass in xvalues_ }

#y_theo = [s.sigma for s in samples.itervalues() ]
y_th_xsec = collections.OrderedDict(sorted(y_theo.items()))

#print "theory xsec: ", y_th_xsec
#print y_th_xsec.values()
#print l.v

y_xsec_vals = array('f', [l.v[i]*y_th_xsec.values()[i] for i in xrange(len(l.v) ) ])

y_xsec_obs_vals = array('f', [l.o[i]*y_th_xsec.values()[i] for i in xrange(len(l.o) ) ])

y_th_xsec_vals = array('f', [th for th in y_th_xsec.itervalues()])

y_bars_d1 =  array('f', ebar_d1)
y_bars_d2 =  array('f', ebar_d2)
y_bars_u1 =  array('f', ebar_u1)
y_bars_u2 =  array('f', ebar_u2)

if theo:
    y_bars_d1 =  array('f', [ebar_d1[i]*y_th_xsec.values()[i] for i in xrange(len(l.v) ) ])
    y_bars_d2 =  array('f', [ebar_d2[i]*y_th_xsec.values()[i] for i in xrange(len(l.v) ) ])
    y_bars_u1 =  array('f', [ebar_u1[i]*y_th_xsec.values()[i] for i in xrange(len(l.v) ) ])
    y_bars_u2 =  array('f', [ebar_u2[i]*y_th_xsec.values()[i] for i in xrange(len(l.v) ) ])

#print "Error bars d1: ", ebar_d1
#print "Error bars xsec d1: ", y_bars_d1

#print "Error bars d2: ", ebar_d2
#print "Error bars xsec d2: ", y_bars_d2

#print "Error bars u1: ", ebar_u1
#print "Error bars xsec u1: ", y_bars_u1

#print "Error bars u2: ", ebar_u2
#print "Error bars xsec u2: ", y_bars_u2


x_bars_1 = array('f', [0]*len(l.v))
x_bars_2 = array('f', [0]*len(l.v))

# need to pick up 
#y_theo = [ s.sigma for s in samples]

median = ROOT.TGraph( len(l.v), xvalues , med_values)
if theo: median = ROOT.TGraph( len(l.v), xvalues , y_xsec_vals)
median.SetLineWidth(2);
median.SetLineStyle(2);
median.SetLineColor(ROOT.kBlue);
median.SetFillColor(0);
median.GetXaxis().SetRangeUser(110, 150);

obs = ROOT.TGraph( len(l.v), xvalues , obs_values)
if theo: obs = ROOT.TGraph( len(l.o), xvalues , y_xsec_obs_vals)
obs.SetLineWidth(2);
obs.SetLineStyle(1);
obs.SetLineColor(ROOT.kBlue);
obs.SetFillColor(0);
obs.GetXaxis().SetRangeUser(110, 150);

theory = ROOT.TGraph( len(l.v), xvalues , y_th_xsec_vals);
theory.SetLineWidth(2);
theory.SetLineStyle(1);
theory.SetLineColor(ROOT.kRed);
theory.SetFillColor(ROOT.kWhite);

band_1sigma = ROOT.TGraphAsymmErrors(len(l.v), xvalues, med_values, x_bars_1, x_bars_2, y_bars_d1, y_bars_u1)
if theo: band_1sigma = ROOT.TGraphAsymmErrors(len(l.v), xvalues, y_xsec_vals, x_bars_1, x_bars_2, y_bars_d1, y_bars_u1)
#band_1sigma = ROOT.TGraphAsymmErrors(len(l.v), masses_value, med_values, 0, 0, 0, 0)
band_1sigma.SetFillColor(ROOT.kGreen + 1)
band_1sigma.SetLineColor(ROOT.kGreen + 1)
band_1sigma.SetMarkerColor(ROOT.kGreen + 1)

band_2sigma = ROOT.TGraphAsymmErrors(len(l.v), xvalues, med_values, x_bars_1, x_bars_2, y_bars_d2, y_bars_u2)
if theo: band_2sigma = ROOT.TGraphAsymmErrors(len(l.v), xvalues,  y_xsec_vals, x_bars_1, x_bars_2, y_bars_d2, y_bars_u2)
band_2sigma.SetTitle("")
band_2sigma.SetFillColor(ROOT.kOrange)
band_2sigma.SetLineColor(ROOT.kOrange)
band_2sigma.SetMarkerColor(ROOT.kOrange)
band_2sigma.GetXaxis().SetTitle("#it{m}_{Z'} [GeV]");
band_2sigma.GetXaxis().SetTitleOffset(0.80);
band_2sigma.GetXaxis().SetLabelSize(0.037);
band_2sigma.GetXaxis().SetTitleSize(0.049);

band_2sigma.GetYaxis().SetTitle("#sigma #times BR [pb]");
band_2sigma.GetYaxis().SetTitleOffset(0.75);
band_2sigma.GetYaxis().SetTitleSize(0.054);
band_2sigma.GetYaxis().SetLabelSize(0.041);
band_2sigma.GetYaxis().SetTitleFont(42);



legend = ROOT.TLegend(0.485,0.5,0.93,0.90)
legend.SetTextSize(0.039);
legend.SetFillStyle(0);
legend.SetBorderSize(0);
legend.SetHeader("95% CL upper limits");
# uncomment for ob as well
# legend.AddEntry(m_y_lineObs_graph,"Observed","ex0p");
# legend.AddEntry(m_y_lineSI_graph,"CL_{S} sign. injected");
if(unblind): legend.AddEntry(obs,"Median observed");
legend.AddEntry(median,"Median expected");
legend.AddEntry(band_1sigma,"68% expected");
legend.AddEntry(band_2sigma,"95% expected");
legend.AddEntry(theory, "Theoretical");
# m_legend.AddEntry(theo, "Theoretical #sigma");
legend.SetEntrySeparation(0.3);
legend.SetFillColor(0);



c1 = ROOT.TCanvas()
ROOT.SetOwnership(c1, False)
c1.cd()
c1.SetGrid() 
c1.SetLogy(1)
band_2sigma.GetXaxis().SetRangeUser(600, 4400)
c1.Update()
band_2sigma.Draw("A3")
band_1sigma.Draw("3")
band_1sigma.SetMaximum(200)
median.Draw("L")
if(unblind):obs.Draw("L")
theory.Draw("L same");
legend.Draw("Same");



lumiTextSize     = 0.6;
lumiTextOffset   = 0.2;
# float cmsTextSize      = 0.75;
# float cmsTextOffset    = 0.1;  // only used in outOfFrame version




pad = ROOT.TPad("pad","pad",0, 0. , 1, 1.);
ROOT.SetOwnership(pad, False)
pad.SetBorderMode(0);

pad.SetTickx(0);
pad.SetTicky(0);
# pad.Draw();
#pad.cd();
t = pad.GetTopMargin();
r = pad.GetRightMargin();
cmsText     = ROOT.TString("CMS")
cmsTextFont   = 61; 
cmsTextSize      = 0.75;

#lumiText = ROOT.TString("41.5 fb^{-1} (13 TeV)")
lumiText = "77.45 fb^{-1} (13 TeV)"
if (opt.year == "2016"):lumiText = "35.92 fb^{-1} (13 TeV)"
elif (opt.year == "2017"):lumiText = "41.53 fb^{-1} (13 TeV)"
elif (opt.year == "2018"):lumiText = "59.8 fb^{-1} (13 TeV)"
elif (opt.year == "Run2"):lumiText = "137.19 fb^{-1} (13 TeV)"
latex_lumi = ROOT.TLatex();
latex_lumi.SetNDC();
latex_lumi.SetTextAngle(0);
latex_lumi.SetTextColor(ROOT.kBlack);

latex_lumi.SetTextFont(42);
latex_lumi.SetTextAlign(31);
latex_lumi.SetTextSize(lumiTextSize*t*0.55);
latex_lumi.DrawLatex(1-r,1-t+lumiTextOffset*t,lumiText);

#latex_cms = ROOT.TLatex()
#latex_cms.SetTextFont(cmsTextFont);
#latex_cms.SetTextSize(cmsTextSize*t);
#latex_cms.SetTextAlign(11);

#l_ch = ROOT.TLatex();
#l_ch.SetNDC();
#l_ch.SetTextSize(lumiTextSize*t*0.71);
#l_ch.SetTextFont(42);
#l_ch.SetTextAlign(11);

l = ROOT.TLatex();
l.SetNDC();
l.SetTextSize(lumiTextSize*t*0.8);
l.SetTextAlign(11); # align left
# l.DrawLatex(0.13,1-t+lumiTextOffset*t*,"CMS");


l.DrawLatex(0.15,0.83,"CMS");

l_preliminary = ROOT.TLatex();
l_preliminary.SetNDC();
l_preliminary.SetTextAlign(31); # align right
l_preliminary.SetTextSize(lumiTextSize*t*0.65);
l_preliminary.SetTextFont(52);
l_preliminary.SetTextAlign(11); # align left
l_preliminary.DrawLatex(0.17,1-t+lumiTextOffset*t,"Work in progress");
l_preliminary.DrawLatex(0.13, 0.81,"");

#l_label = ROOT.TLatex();
#l_label.SetNDC();
#l_label.SetTextAlign(31); # align right
#l_label.SetTextSize(lumiTextSize*t*0.71); #0.5
#l_label.SetTextFont(42);
#l_label.SetTextAlign(11);


c1.Update()
c1.SaveAs("plots/test_limitPlot_%s_%s.pdf" % (opt.year, cat))
