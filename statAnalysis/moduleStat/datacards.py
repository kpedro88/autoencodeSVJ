import ROOT
import json

from ROOT import RooRealVar, RooDataHist, RooArgList, RooGenericPdf, RooExtendPdf, RooWorkspace, RooFit, RooDataSet, \
    RooArgSet, RooCategory, RooFitResult, RooCurve
import os, sys
from array import array
import copy, math, pickle

import numpy as np
from numpy import ndarray

from moduleStat.settings import *

ROOT.TH1.SetDefaultSumw2()
ROOT.TH1.AddDirectory(False)
ROOT.gROOT.SetStyle('Plain')
ROOT.gROOT.SetBatch()
ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(0)
# ROOT.Math.MinimizerOptions.SetDefaultTolerance(1e-3);
# ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-8)
# *****************************************************
#                                                       #
#   getRate(process, ifile)                             #
#                                                       #
#   getCard(sig, ch, ifilename, outdir)                 #
#                                                       #
# *******************************************************#


# *******************************************************#
#                                                       #
#                     Utility Functions                 #
#                                                       #
# *******************************************************#

isData = False


def getRate(ch, process, ifile):
    hName = ch + "/" + process
    
    h = ifile.Get(hName)
    
    if type(h) is not ROOT.TH1F:
        print("ERROR -- invalid hist type, input histogram probably not in the input file")
        print("\thist name: ", hName)
        exit(0)
    
    return h.Integral(1, h.GetXaxis().GetNbins() - 1)


def getHist(ch, process, ifile):
    hName = ch + "/" + process
    print("Histo Name ", hName)
    h = ifile.Get(hName)
    h.SetDirectory(0)
    return h


def getEfficiency(sig, channels, ifilename):
    ifile = None
    try:
        ifile = ROOT.TFile.Open(ifilename)
    except IOError:
        print("Cannot open ", ifilename)
    else:
        print("Opening file ", ifilename)
        ifile.cd()
    
    rates = {}
    for ch in channels:
        rates[ch] = getRate(ch, sig, ifile)
    
    sumjets = sum(iter(rates.values())) * 2
    print(sumjets)
    
    evts2SVJ = sum(i for ch, i in iter(rates.items()) if "SVJ2" in ch)
    print(evts2SVJ)
    evts1SVJ = sum(i for ch, i in iter(rates.items()) if "SVJ1" in ch)
    
    num = evts1SVJ + (2 * evts2SVJ)
    
    eff = num / sumjets
    return eff


def getRSS(sig, ch, variable, model, dataset, fitRes, norm=-1, label="nom"):
    name = model.GetName()
    order = int(name[-1])
    npar = fitRes[0].floatParsFinal().getSize() if len(fitRes) > 0 else 0
    varArg = ROOT.RooArgSet(variable)
    
    # frame = variable.frame()
    frame = variable.frame(ROOT.RooFit.Title(""))
    dataset.plotOn(frame, RooFit.Invisible())
    if len(fitRes) > 0: model.plotOn(frame, RooFit.VisualizeError(fitRes[0], 1, False),
                                     RooFit.Normalization(norm if norm > 0 else dataset.sumEntries(),
                                                          ROOT.RooAbsReal.NumEvent), RooFit.LineColor(ROOT.kBlue),
                                     RooFit.FillColor(ROOT.kOrange), RooFit.FillStyle(1001), RooFit.DrawOption("FL"),
                                     RooFit.Range("Full"))
    if len(fitRes) > 0:
        graphFit = model.plotOn(frame, RooFit.VisualizeError(fitRes[0], 1, False),
                                RooFit.Normalization(norm if norm > 0 else dataset.sumEntries(),
                                                     ROOT.RooAbsReal.NumEvent), RooFit.LineColor(ROOT.kBlue),
                                RooFit.FillColor(ROOT.kOrange), RooFit.FillStyle(1001), RooFit.DrawOption("FL"),
                                RooFit.Range("Full"))
    
    model.plotOn(frame, RooFit.Normalization(norm if norm > 0 else dataset.sumEntries(), ROOT.RooAbsReal.NumEvent),
                 RooFit.LineColor(ROOT.kBlue), RooFit.FillColor(ROOT.kOrange), RooFit.FillStyle(1001),
                 RooFit.DrawOption("L"), RooFit.Name(model.GetName()), RooFit.Range("Full"))
    model.paramOn(frame, RooFit.Label(model.GetTitle()), RooFit.Layout(0.45, 0.95, 0.94), RooFit.Format("NEAU"))
    
    graphData = dataset.plotOn(frame, RooFit.DataError(ROOT.RooAbsData.Poisson if isData else ROOT.RooAbsData.SumW2),
                               RooFit.DrawOption("PE0"), RooFit.Name(dataset.GetName()))
    
    # (ROOT.TVirtualFitter.GetFitter()).GetConfidenceIntervals(model)
    
    # f1 = ROOT.TF1(model)
    # f1.SetName("f")
    # f2 = ROOT.TF1(model)
    # f2.SetName("g")
    # nPar = f2.GetNpar()
    # errorpar = np.zeros(nPar)
    
    # f2.SetParErrors()
    # f3 = ROOT.TF1("error", "f/g")
    
    pulls = frame.pullHist(dataset.GetName(), model.GetName(), True)
    residuals = frame.residHist(dataset.GetName(), model.GetName(), False, True)  # this is y_i - f(x_i)
    
    roochi2 = frame.chiSquare()  # dataset.GetName(), model.GetName()) #model.GetName(), dataset.GetName()
    print("forcing bins: 23")
    nbins = 23
    chi = roochi2 * (nbins - order)
    print("pls: ", chi, nbins)
    roopro = ROOT.TMath.Prob(chi, nbins - order)
    
    frame.SetMaximum(frame.GetMaximum() * 1.2)
    frame.SetMinimum(0.01)
    print("==========> len(sig): ", len(sig))
    length = 1
    if (length < 2):
        c = ROOT.TCanvas("c_" + ch + model.GetName(), ch, 800, 800)
        
        c.cd()
        pad1 = ROOT.TPad("pad1", "pad1", 0., 0.35, 1., 1.0)
        ROOT.SetOwnership(pad1, False)
        pad1.SetBottomMargin(0.)
        pad1.SetGridx()
        pad1.SetGridy()
        pad1.SetLogy()
        pad1.Draw()
        pad1.cd()
        
        frame.Draw()
        frame.SetTitle("")
        
        # txt = ROOT.TText(2500, 1000, "chiSquared: " + str(roochi2))
        # txt = ROOT.TText(2500, 6., "chiSquared: " + str(roochi2))
        roochi2_small = format(roochi2, '.2f')
        txt = ROOT.TText(0.65, 0.7, "chiSquared: " + str(roochi2_small))
        txt.SetNDC()
        txt.SetTextSize(0.04)
        txt.SetTextColor(ROOT.kRed)
        txt.Draw();
        
        # txt_2 = ROOT.TText(2500, 800, "Prob: " + str(roopro))
        # txt_2 = ROOT.TText(2500, 2., "Prob: " + str(roopro))
        roopro_small = format(roopro, '.2f')
        txt_2 = ROOT.TText(0.65, 0.65, "prob: " + str(roopro_small))
        txt_2.SetNDC()
        txt_2.SetTextSize(0.04)
        txt_2.SetTextColor(ROOT.kRed)
        txt_2.Draw();
        
        c.cd()
        c.Update()
        c.cd()
        pad2 = ROOT.TPad("pad2", "pad2", 0, 0.1, 1, 0.3)
        ROOT.SetOwnership(pad2, False)
        pad2.SetTopMargin(0);
        pad2.SetBottomMargin(0.25);
        pad2.SetGridx();
        pad2.SetGridy();
        pad2.Draw();
        pad2.cd()
        pad2.Clear()
        # frame.Draw()
        c.Update()
        c.Modified()
        
        frame_res = variable.frame(ROOT.RooFit.Title(""))
        frame_res.addPlotable(pulls, "P")
        frame_res.GetYaxis().SetRangeUser(-3.5, 3.5)
        frame_res.SetTitle("")
        frame_res.GetYaxis().SetTitle("(N^{data}-f(i))/#sigma")
        frame_res.GetYaxis().SetTitleOffset(.3)
        frame_res.GetXaxis().SetTitleOffset(1.)
        frame_res.GetYaxis().SetTitleSize(.13)
        frame_res.GetXaxis().SetTitle("m_{T} (GeV)")
        frame_res.GetXaxis().SetTitleSize(.13)
        frame_res.GetXaxis().SetLabelSize(.12)
        frame_res.GetYaxis().SetLabelSize(.12)
        frame_res.Draw("e0SAME")
        frame_res.SetTitle("")
    
    hist = graphData.getHist()
    xmin_ = graphData.GetXaxis().GetXmin()
    xmax_ = graphData.GetXaxis().GetXmax()
    
    if (length < 2):
        c.Update()
        c.Range(xmin_, -3.5, xmax_, 3.5)
        line = ROOT.TLine(xmin_, 0., xmax_, 0.)
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(2)
        line.Draw("same")
        
        c.SaveAs("Residuals/Residuals_" + ch + "_" + name + "_log.pdf")
        '''
        c2 = ROOT.TCanvas("c2_"+ch+model.GetName(), ch, 800, 800)
        c2.cd()
        pad1_2 = ROOT.TPad("pad1_2", "pad1_2", 0., 0.35, 1., 1.0)
        ROOT.SetOwnership(pad1_2, False)
        pad1_2.SetBottomMargin(0.);
        pad1_2.SetGridx();
        pad1_2.SetGridy();
        pad1_2.SetLogy()
        pad1_2.Draw()
        pad1_2.cd()
        frame.Draw()
        
        txt = ROOT.TText(3000, 4., "chiSquared: " + str(roochi2))
        txt.SetTextSize(0.04)
        txt.SetTextColor(ROOT.kRed)
        txt.Draw()
        
        c2.cd()
        c2.Update()
        c2.cd()
        pad2_2 = ROOT.TPad("pad2_2", "pad2_2", 0, 0.05, 1, 0.3)
        ROOT.SetOwnership(pad2_2, False)
        pad2_2.SetTopMargin(0);
        pad2_2.SetBottomMargin(0.1);
        pad2_2.SetGridx();
        pad2_2.SetGridy();
        pad2_2.Draw();
        pad2_2.cd()
        pad2_2.Clear()
 #frame.Draw()
        c2.Update()
        c2.Modified()

        '''
    # calculate RSS
    res, rss, chi1, chi2 = 0, 0., 0., 0.,
    
    xmin, xmax = array('d', [0.]), array('d', [0.])
    dataset.getRange(variable, xmin, xmax)
    #       print "N bins: ", hist.GetN()
    # nBins = graphData.GetNbinsX()
    ratioHist = ROOT.TH1F("RatioPlot", "Ratio Plot", hist.GetN(), xmin_, xmax_)
    ROOT.SetOwnership(ratioHist, False)
    
    #      ratioHist.SetBins(nBins, graphData.GetXaxis().GetXbins().GetArray())
    
    hist.Dump()
    ratioHist.Dump()
    ratioHist.Print('all')
    
    sumErrors = 0
    
    fitmodel = graphFit.getHist()
    # fitmodel.Print()
    fitmodel_curve = graphFit.getCurve("Bkg_" + str(ch) + str(order) + "_Norm[mH]_errorband")
    # print "fitmodel_curve: "
    # fitmodel_curve.Print()
    
    # print "check curve bins"
    # print np.ndarray(fitmodel_curve.GetN(), 'd', fitmodel_curve.GetX())
    c_x = np.ndarray(fitmodel_curve.GetN(), 'd', fitmodel_curve.GetX())
    c_y_all = np.ndarray(fitmodel_curve.GetN(), 'd', fitmodel_curve.GetY())
    c_y_up = c_y_all[0:len(c_y_all) / 2]
    c_y_down = list(reversed(c_y_all[len(c_y_all) / 2 + 1:]))
    
    # print c_y_up
    # print c_y_down
    
    def findClosestLowerBin(x, bins):
        for ib in range(len(bins)):
            if bins[ib] > x:
                return ib - 1
    
    # print fitmodel_curve.GetY()[0], fitmodel_curve.GetY()[1], fitmodel_curve.GetY()[2]
    # print  fitmodel_curve.GetY()[fitmodel_curve.GetN()/2],  fitmodel_curve.GetY()[fitmodel_curve.GetN()/2+1],  fitmodel_curve.GetY()[fitmodel_curve.GetN()/2+2],
    
    for i in range(0, hist.GetN()):
        
        print("bin x and y: ", hist.GetX()[i], hist.GetY()[i])
        print(hist.GetN(), fitmodel_curve.GetN())
        #              histBin = fitmodel_curve.FindBin(hist.GetX()[i])
        #              print "curve bin x and y: ",  fitmodel_curve.GetX()[i], fitmodel_curve.GetX()[histBin], fitmodel_curve.GetBinContent(histBin),  fitmodel_curve.GetBinContent(histBin+ fitmodel_curve.GetN()/2)
        err_up = c_y_up[findClosestLowerBin(hist.GetX()[i], c_x)]
        err_down = c_y_down[findClosestLowerBin(hist.GetX()[i], c_x)]
        
        print("x, y, eyup, eydo: ", hist.GetX()[i], hist.GetY()[i], err_up, err_down)
        if (hist.GetY()[i] > 0):
            diffErrors = abs(err_up - err_down) / hist.GetY()[i]
        else:
            diffErrors = 0
        sumErrors += diffErrors
        
        hx, hy = hist.GetX()[i], hist.GetY()[i]
        hey = hist.GetErrorY(i)
        hexlo, hexhi = hist.GetErrorXlow(i), hist.GetErrorXhigh(i)
        ry = residuals.GetY()[i]
        pull = pulls.GetY()[i]
        
        if (hx - hexlo) > xmax[0] and hx + hexhi > xmax[0]:
            continue
        
        if hy <= 0.:
            continue
        
        res += ry
        # print "residuals ", ry
        rss += ry ** 2
        
        chi1 += abs(pull)
        chi2 += pull * 2
        
        # print " Residual: ", ry, "Bin Content: ", hy, " new bin: ", (ry-hy)/(-1*hy)
        
        ratioHist.SetBinContent(i + 1, (ry - hy) / (-1 * hy))
        
        ratioHist.SetBinError(i + 1, (hey / hy ** 2))
        ratioHist.Print('all')
    
    # if hist.GetX()[i] - hist.GetErrorXlow(i) > xmax[0] and hist.GetX()[i] + hist.GetErrorXhigh(i) > xmax[0]: continue# and abs(pulls.GetY()[i]) < 5:
    # if hist.GetY()[i] <= 0.: continue
    # res += residuals.GetY()[i]
    # print "residuals ", residuals.GetY()[i]
    # rss += residuals.GetY()[i]**2
    
    # print pulls.GetY()[i]
    # chi1 += abs(pulls.GetY()[i])
    # chi2 += pulls.GetY()[i]**2
    # print " Residual: ", residuals.GetY()[i], " Bin Content: ", hist.GetY()[i], " new bin: ", (residuals.GetY()[i] - hist.GetY()[i])/(-1 * hist.GetY()[i])
    # ratioHist.SetBinContent(i, (residuals.GetY()[i] - hist.GetY()[i])/(-1*hist.GetY()[i]))
    # print "ERROR: ", hist.GetHistogram().GetBinError(i)
    # ratioHist.SetBinError(i, (hist.GetHistogram().GetBinError(i)/ pow(hist.GetY()[i],2)))
    
    print("=======> sumErrors: ", sumErrors)
    
    rss = math.sqrt(rss)
    out = {"chi2": chi2, "chi1": chi1, "rss": rss, "res": res, "nbins": hist.GetN(), "npar": npar}
    # c.SaveAs(carddir + "/plots/Residuals_"+ch+"_"+name+".pdf")
    
    if (length < 2):
        c2 = ROOT.TCanvas("c2_" + ch + model.GetName(), ch, 800, 800)
        c2.cd()
        pad1_2 = ROOT.TPad("pad1_2", "pad1_2", 0., 0.35, 1., 1.0)
        ROOT.SetOwnership(pad1_2, False)
        pad1_2.SetBottomMargin(0.);
        pad1_2.SetGridx();
        pad1_2.SetGridy();
        pad1_2.SetLogy()
        pad1_2.Draw()
        pad1_2.cd()
        frame.Draw()
        frame.SetTitle("")
        
        # txt = ROOT.TText(2500, 6., "chiSquared: " + str(roochi2))
        # txt = ROOT.TText(2500, 1000, "chiSquared: " + str(roochi2))
        txt = ROOT.TText(0.65, 0.7, "chiSquared: " + str(roochi2_small))
        txt.SetNDC()
        txt.SetTextSize(0.04)
        txt.SetTextColor(ROOT.kRed)
        txt.Draw()
        
        # txt_2 = ROOT.TText(2500, 800, "Prob: " + str(roopro))
        txt_2 = ROOT.TText(0.65, 0.65, "prob: " + str(roopro_small))
        txt_2.SetNDC()
        # txt_2 = ROOT.TText(2500, 2., "Prob: " + str(roopro))
        txt_2.SetTextSize(0.04)
        txt_2.SetTextColor(ROOT.kRed)
        txt_2.Draw();
        
        #              txt2 = ROOT.TText(3000, 1., "Error: "+  str(sumErrors))
        #              txt2.SetTextSize(0.04)
        #              txt2.SetTextColor(ROOT.kOrange)
        # txt2.Draw("same");
        
        c2.cd()
        c2.Update()
        c2.cd()
        pad2_2 = ROOT.TPad("pad2_2", "pad2_2", 0, 0.05, 1, 0.3)
        ROOT.SetOwnership(pad2_2, False)
        pad2_2.SetTopMargin(0);
        pad2_2.SetBottomMargin(0.25);
        pad2_2.SetGridx();
        pad2_2.SetGridy();
        pad2_2.Draw();
        pad2_2.cd()
        pad2_2.Clear()
        # frame.Draw()
        c2.Update()
        c2.Modified()
        frame.SetTitle("")
        
        ratioHist.GetYaxis().SetRangeUser(0., 2.)
        ratioHist.SetTitle("")
        ratioHist.GetYaxis().SetTitle("f(i)/N^{data}")
        ratioHist.GetXaxis().SetTitle("m_{T} (GeV)")
        ratioHist.GetYaxis().SetTitleSize(.13)
        ratioHist.GetYaxis().SetTitleOffset(.3)
        ratioHist.GetXaxis().SetTitleOffset(1.)
        ratioHist.GetXaxis().SetTitleSize(.13)
        ratioHist.GetXaxis().SetLabelSize(.12)
        ratioHist.GetYaxis().SetLabelSize(.12)
        
        # ratioHist.SetMarkerSize(2)
        ratioHist.SetMarkerStyle(20)
        ratioHist.Draw("PE")
        line2 = ROOT.TLine(xmin_, 1., xmax_, 1.)
        line2.SetLineColor(ROOT.kRed)
        line2.SetLineWidth(2)
        line2.Draw("same")
        frame.SetTitle("")
        
        c2.SaveAs("Residuals/Residuals_" + ch + "_" + name + "_ratio_log.pdf")
    
    return out


def fisherTest(RSS1, RSS2, o1, o2, N):
    print("Testing functions with parameters o1 ", o1, " and o2 ", o2, " with RSS RSS1 ", RSS1, " and RSS2 ", RSS2,
          " and N ", N)
    n1 = o1
    n2 = o2
    F = ((RSS1 - RSS2) / (n2 - n1)) / (RSS2 / (N - n2))
    print("***************** Value of F: ", F)
    CL = 1. - ROOT.TMath.FDistI(F, n2 - n1, N - n2)
    return CL


# *******************************************************#
#                                                       #
#                      Datacard                         #
#                                                       #
# *******************************************************#

def getCard(sig, ch, ifilename, outdir, doModelling, mode="histo", bias=False, verbose=False):
    ### Cmpute signal efficiencies
    
    effname = "Efficiencies.txt"
    efile = open(effname, 'r')
    effline = efile.readline()
    
    effs = json.loads(effline)
    
    eff = effs[sig]
    
    try:
        ifile = ROOT.TFile.Open(ifilename)
    except IOError:
        print("Cannot open ", ifilename)
    else:
        print("Opening file ", ifilename)
        ifile.cd()
    
    print("BIAS?", bias)
    workdir_ = ifilename.split("/")[:-1]
    WORKDIR = "/".join(workdir_) + "/"
    carddir = outdir + "/" + sig + "/"
    if not os.path.isdir(outdir): os.system('mkdir ' + outdir)
    if not os.path.isdir(outdir + "/" + sig): os.system('mkdir ' + carddir)
    if not os.path.isdir(outdir + carddir + "plots/"): os.system('mkdir ' + carddir + "plots/")
    if not os.path.isdir(outdir + carddir + "Fisher/"): os.system('mkdir ' + carddir + "Fisher/")
    
    hist_filename = os.getcwd() + "/" + ifilename
    hist = getHist(ch, sig, ifile)
    
    # *******************************************************#
    #                                                       #
    #                   Generate workspace                  #
    #                                                       #
    # *******************************************************#

    parNames = []
    
    if (mode == "template"):
        
        ###ATT: CHECK BKG AND DATA NORMALIZATION AND DISTRIBUTION
        histBkgData = getHist(ch, "Bkg", ifile)
        histData = getHist(ch, "data_obs", ifile)
        print("channel ", ch)
        print("channel ", sig)
        
        histSig = getHist(ch, sig, ifile)
        print("histSigData: ", histSig.Integral())
        # print "histBkgData: ", histBkgData.Integral()
        xvarmin = 1500.
        xvarmax = 3800.
        mT = RooRealVar("mH", "m_{T}", xvarmin, xvarmax, "GeV")
        binMax = histData.FindBin(xvarmax)
        bkgData = RooDataHist("bkgdata", "Data (MC Bkg)", RooArgList(mT), histBkgData, 1.)
        obsData = RooDataHist("data_obs", "(pseudo) Data", RooArgList(mT), histData, 1.)
        sigData = RooDataHist("sigdata", "Data (MC sig)", RooArgList(mT), histSig, 1.)
        print("SUM ENTRIES: ", sigData.sumEntries())
        print("Bkg Integral: ", histData.Integral())
        # nBkgEvts = histBkgData.Integral(1, histBkgData.GetXaxis().GetNbins()-1)
        nBkgEvts = histData.Integral(1, binMax)
        nDataEvts = histData.Integral(1, binMax)
        #              nBkgEvts = histData.Integral(1, histData.GetXaxis().GetNbins()-1)
        #              nDataEvts = histData.Integral(1, histData.GetXaxis().GetNbins()-1)
        # print "Bkg Events: ", nBkgEvts
        
        print("channel: ", ch)
        normBkg = RooRealVar("Bkg_" + ch + "_norm", "Number of background events", nBkgEvts, 0., 2.e4)
        ch_red = ch[:-5]
        modelName = "Bkg_" + ch
        modelAltName = "Bkg_Alt_" + ch

   
        
        if (doModelling):
            # ch_red = ch
            print("channel: ", ch_red)
            p1_1 = RooRealVar(ch_red + "_p1_1", "p1", 1., -1000., 1000.)
            p1_2 = RooRealVar(ch_red + "_p1_2", "p1", 1., -1000., 1000.)
            if (ch == "lowSVJ2_2017" or ch == "highSVJ2_2017"):
                p1_3 = RooRealVar(ch_red + "_p1_3", "p1", 1., -10., 10.)
            else:
                p1_3 = RooRealVar(ch_red + "_p1_3", "p1", 1., -1000., 1000.)
            p1_4 = RooRealVar(ch_red + "_p1_4", "p1", 1., -1000., 1000.)
            p2_2 = RooRealVar(ch_red + "_p2_2", "p2", 1., -1000., 1000.)
            if (
                    ch == "highSVJ2_2016" or ch == "highSVJ2_2017" or ch == "highSVJ2_2018" or ch == "lowSVJ2_2017" or ch == "highSVJ"):
                p2_3 = RooRealVar(ch_red + "_p2_3", "p2", 1., -10., 10.)
            else:
                p2_3 = RooRealVar(ch_red + "_p2_3", "p2", 1., -1000., 1000.)
            p2_4 = RooRealVar(ch_red + "_p2_4", "p2", 1., -1000., 1000.)
            if (
                    ch == "highSVJ2_2016" or ch == "highSVJ2_2017" or ch == "highSVJ2_2018" or ch == "lowSVJ2_2017" or ch == "highSVJ2"):
                p3_3 = RooRealVar(ch_red + "_p3_3", "p3", 1., -10., 10.)
            else:
                p3_3 = RooRealVar(ch_red + "_p3_3", "p3", 1., -1000., 1000.)
            p3_4 = RooRealVar(ch_red + "_p3_4", "p3", 1., -1000., 1000.)
            p4_4 = RooRealVar(ch_red + "_p4_4", "p4", 1., -1000., 1000.)
            
            # modelBkg1 = RooAbsPdf(modelName+"1", "pow(1 + mT/13000, p1_1)", RooArgSet(mT, p1_1))
            modelBkg1 = RooGenericPdf(modelName + "1", "Bkg. fit (1 par.)", "pow(1 + @0/13000, @1) ",
                                      RooArgList(mT, p1_1))
            modelBkg2 = RooGenericPdf(modelName + "2", "Bkg. fit (2 par.)",
                                      "pow(1 + @0/13000, @1) * pow(@0/13000, -@2)", RooArgList(mT, p1_2, p2_2))
            modelBkg3 = RooGenericPdf(modelName + "3", "Bkg. fit (3 par.)",
                                      "pow(1 + @0/13000, @1) * pow(@0/13000, -@2-@3*log(@0/13000))",
                                      RooArgList(mT, p1_3, p2_3, p3_3))
            modelBkg4 = RooGenericPdf(modelName + "4", "Bkg. fit (4 par.)",
                                      "pow(1 + @0/13000, @1) * pow(@0/13000, -@2-log(@0/13000)*(@3 + @4* log(@0/13000)))",
                                      RooArgList(mT, p1_4, p2_4, p3_4, p4_4))
            
            RSS = {}
            fitrange = "Full"
            # if (ch == "highSVJ1_2016"): fitrange = "Low,High"
            
            fitRes1 = modelBkg1.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                      RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(2),
                                      RooFit.Range(fitrange))
            fitRes2 = modelBkg2.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                      RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(2),
                                      RooFit.Range(fitrange))
            fitRes3 = modelBkg3.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                      RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(2),
                                      RooFit.Range(fitrange))
            fitRes4 = modelBkg4.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                      RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(2),
                                      RooFit.Range(fitrange))
            
            # r = RooFitResult(fitRes1)
            # r.Print()
            # chi2s_bkg1 = modelBkg1.createChi2(obsData).getVal()
            # chi2s_bkg2 = modelBkg2.createChi2(obsData).getVal()
            # chi2s_bkg3 = modelBkg3.createChi2(obsData).getVal()
            # chi2s_bkg4 = modelBkg4.createChi2(obsData).getVal()
            # print "ChiSquares:", chi2s_bkg1, " ", chi2s_bkg2, " ", chi2s_bkg3, " ", chi2s_bkg4
            
            # chi2s_bkg1 = frame->chiSquare(nFloatParam) ;
            
            xframe = mT.frame(ROOT.RooFit.Title("extended ML fit example"))
            
            c1 = ROOT.TCanvas()
            c1.cd()
            obsData.plotOn(xframe, RooFit.Name("obsData"))
            
            #              modelBkg1.plotOn(xframe,ROOT.RooFit.LineColor(ROOT.kGreen))
            modelBkg1.plotOn(xframe, RooFit.Name("modelBkg1"), ROOT.RooFit.LineColor(ROOT.kPink + 6),
                             RooFit.Range("Full"))
            modelBkg2.plotOn(xframe, RooFit.Name("modelBkg2"), ROOT.RooFit.LineColor(ROOT.kBlue - 4),
                             RooFit.Range("Full"))
            modelBkg3.plotOn(xframe, RooFit.Name("modelBkg3"), ROOT.RooFit.LineColor(ROOT.kRed - 4),
                             RooFit.Range("Full"))
            modelBkg4.plotOn(xframe, RooFit.Name("modelBkg4"), ROOT.RooFit.LineColor(ROOT.kGreen + 1),
                             RooFit.Range("Full"))
            
            chi2s_bkg1 = xframe.chiSquare("modelBkg1", "obsData")
            chi2s_bkg2 = xframe.chiSquare("modelBkg2", "obsData")
            chi2s_bkg3 = xframe.chiSquare("modelBkg3", "obsData")
            chi2s_bkg4 = xframe.chiSquare("modelBkg4", "obsData")
            
            xframe.SetMinimum(0.0002)
            xframe.Draw()
            
            txt1 = ROOT.TText(3000., 4., str(chi2s_bkg1))
            txt1.SetTextSize(0.04)
            txt1.SetTextColor(ROOT.kPink + 6)
            xframe.addObject(txt1)
            txt2 = ROOT.TText(3000., 3.5, str(chi2s_bkg2))
            txt2.SetTextSize(0.04)
            txt2.SetTextColor(ROOT.kBlue - 4)
            xframe.addObject(txt2)
            txt3 = ROOT.TText(3000., 3.0, str(chi2s_bkg3))
            txt3.SetTextSize(0.04)
            txt3.SetTextColor(ROOT.kRed - 4)
            xframe.addObject(txt3)
            txt4 = ROOT.TText(3000., 2.5, str(chi2s_bkg4))
            txt4.SetTextSize(0.04)
            txt4.SetTextColor(ROOT.kGreen + 1)
            xframe.addObject(txt4)
            txt1.Draw()
            txt2.Draw()
            txt3.Draw()
            txt4.Draw()
            
            c1.SetLogy()
            c1.SaveAs(carddir + "/plots/TestAfterFit_" + ch + ".pdf")
            
            RSS[1] = getRSS(sig, ch, mT, modelBkg1, obsData, [fitRes1], nBkgEvts)
            RSS[2] = getRSS(sig, ch, mT, modelBkg2, obsData, [fitRes2], nBkgEvts)
            RSS[3] = getRSS(sig, ch, mT, modelBkg3, obsData, [fitRes3], nBkgEvts)
            RSS[4] = getRSS(sig, ch, mT, modelBkg4, obsData, [fitRes4], nBkgEvts)
            
            print(RSS[2])
            
            # **********************************************************
            #                    ALTERNATIVE MODEL                    *
            # **********************************************************
            if bias:
                
                # alternative function is Silvio's nominal function, but with +1 instead of -1
                normAlt = RooRealVar("Bkg_" + ch + "alt_norm", "Number of background events", nBkgEvts, 0., 2.e4)
                # p1_1_alt = RooRealVar(ch + "_p1_1_alt", "p1", 1., -100., 1000.)
                # p1_2_alt = RooRealVar(ch + "_p1_2_alt", "p1", 1., -10, 100.)
                # p1_3_alt = RooRealVar(ch + "_p1_3_alt", "p1", 1., 0., 50.)
                # p1_4_alt = RooRealVar(ch + "_p1_4_alt", "p1", 1., 0., 100.)
                # p2_2_alt = RooRealVar(ch + "_p2_2_alt", "p2", 1., -10., 10.)
                # p2_3_alt = RooRealVar(ch + "_p2_3_alt", "p2", 1., -10., 10.)
                # p2_4_alt = RooRealVar(ch + "_p2_4_alt", "p2", 1., -100., 1000.)
                # p3_3_alt = RooRealVar(ch + "_p3_3_alt", "p3", 1., -10., 100.)
                # p3_4_alt = RooRealVar(ch + "_p3_4_alt", "p3", 1., -10., 100.)
                # p4_4_alt = RooRealVar(ch + "_p4_4_alt", "p4", 1., -10., 100.)
                
                p1_1_alt = RooRealVar(ch + "_p1_1_alt", "p1", 1., -3.4, 1000.)  # -100
                p1_2_alt = RooRealVar(ch + "_p1_2_alt", "p1", 1., -3.4, 100.)  # -100
                p1_3_alt = RooRealVar(ch + "_p1_3_alt", "p1", 1., 0., 100.)  # 0.
                p1_4_alt = RooRealVar(ch + "_p1_4_alt", "p1", 1., 0., 100.)  # 0.
                if (ch == "highSVJ2_2016" or ch == "highSVJ1_2018" or ch == "lowSVJ1_2018" or ch == "lowSVJ2_2018"):
                    p2_2_alt = RooRealVar(ch + "_p2_2_alt", "p2", 1., -10., 10.)
                else:
                    p2_2_alt = RooRealVar(ch + "_p2_2_alt", "p2", 1., -100., 100.)
                p2_3_alt = RooRealVar(ch + "_p2_3_alt", "p2", 1., -10., 10.)
                p2_4_alt = RooRealVar(ch + "_p2_4_alt", "p2", 1., -10., 10.)
                if (
                        ch == "highSVJ2_2016" or ch == "highSVJ1_2017" or ch == "lowSVJ1_2017" or ch == "lowSVJ2_2017" or ch == "highSVJ1_2018" or ch == "lowSVJ1_2018" or ch == "lowSVJ2_2018"):
                    p3_3_alt = RooRealVar(ch + "_p3_3_alt", "p3", 1., 0., 100.)
                else:
                    p3_3_alt = RooRealVar(ch + "_p3_3_alt", "p3", 1., -10., 100.)
                    # p3_4_alt = RooRealVar(ch + "_p3_4_alt", "p3", 1., 0., 100.)
                if (
                        ch == "highSVJ1_2016" or ch == "highSVJ2_2016" or ch == "lowSVJ1_2017" or ch == "lowSVJ2_2017" or ch == "highSVJ2_2017" or ch == "highSVJ2_2018" or ch == "lowSVJ1_2018" or ch == "lowSVJ2_2018"):
                    p3_4_alt = RooRealVar(ch + "_p3_4_alt", "p3", 1., -10., 100.)
                else:
                    p3_4_alt = RooRealVar(ch + "_p3_4_alt", "p3", 1., 0., 100.)
                
                if (
                        ch == "highSVJ1_2016" or ch == "highSVJ2_2016" or ch == "highSVJ2_2017" or ch == "lowSVJ1_2017" or ch == "lowSVJ2_2017" or ch == "highSVJ2_2018" or ch == "lowSVJ1_2018" or ch == "lowSVJ2_2018"):
                    p4_4_alt = RooRealVar(ch + "_p4_4_alt", "p4", 1., 0., 100.)
                else:
                    p4_4_alt = RooRealVar(ch + "_p4_4_alt", "p4", 1., -10., 100.)
                
                modelAlt1 = RooGenericPdf(modelAltName + "1", "Bkg. fit (1 par.)", "(exp(log(@1*(@0/13000) + 1)))",
                                          RooArgList(mT, p1_1_alt))
                modelAlt2 = RooGenericPdf(modelAltName + "2", "Bkg. fit (2 par.)",
                                          "(exp(log(@1*(@0/13000) + 1))) / pow((@0/13000),@2)",
                                          RooArgList(mT, p1_2_alt, p2_2_alt))
                #              modelAlt3 = RooGenericPdf(modelAltName+"3", "Bkg. fit (3 par.)", "(exp(log(@1*(@0/13000) + 1)) + @3 * (@0/13000)) / pow((@0/13000),@2)", RooArgList(mT, p1_3_alt, p2_3_alt, p3_3_alt))
                modelAlt3 = RooGenericPdf(modelAltName + "3", "Bkg. fit (3 par.)",
                                          "(exp(-(log(@1*(@0/13000) + 1) + @3 * (@0/13000)))) / pow((@0/13000),@2)",
                                          RooArgList(mT, p1_3_alt, p2_3_alt, p3_3_alt))
                modelAlt4 = RooGenericPdf(modelAltName + "4", "Bkg. fit (4 par.)",
                                          "(exp(-(log(@1*(@0/13000) + 1) + @3 * (@0/13000) + @4 * pow(@0/13000,2)))) / pow((@0/13000),@2) ",
                                          RooArgList(mT, p1_4_alt, p2_4_alt, p3_4_alt, p4_4_alt))
                
                RSS_alt = {}
                fitrange = "Full"
                # if (ch == "highSVJ1_2016"): fitrange = "Low,High"
                
                fitRes1_alt = modelAlt1.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                              RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(1),
                                              RooFit.Range(fitrange))
                fitRes2_alt = modelAlt2.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                              RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(1),
                                              RooFit.Range(fitrange))
                fitRes3_alt = modelAlt3.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                              RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(1),
                                              RooFit.Range(fitrange))
                fitRes4_alt = modelAlt4.fitTo(obsData, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True),
                                              RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(1),
                                              RooFit.Range(fitrange))
                
                RSS_alt[1] = getRSS(sig, ch, mT, modelAlt1, obsData, [fitRes1_alt], nBkgEvts, label="alt")
                RSS_alt[2] = getRSS(sig, ch, mT, modelAlt2, obsData, [fitRes2_alt], nBkgEvts, label="alt")
                RSS_alt[3] = getRSS(sig, ch, mT, modelAlt3, obsData, [fitRes3_alt], nBkgEvts, label="alt")
                RSS_alt[4] = getRSS(sig, ch, mT, modelAlt4, obsData, [fitRes4_alt], nBkgEvts, label="alt")
                
                if (length < 2):
                    xframeAlt = mT.frame(ROOT.RooFit.Title("extended ML fit example"))
                    
                    c2 = ROOT.TCanvas()
                    c2.cd()
                    obsData.plotOn(xframeAlt)
                    
                    modelAlt1.plotOn(xframeAlt, RooFit.Name("modelAlt1"), ROOT.RooFit.LineColor(ROOT.kPink))
                    modelAlt2.plotOn(xframeAlt, RooFit.Name("modelAlt2"), ROOT.RooFit.LineColor(ROOT.kBlue))
                    modelAlt3.plotOn(xframeAlt, RooFit.Name("modelAlt3"), ROOT.RooFit.LineColor(ROOT.kRed))
                    modelAlt4.plotOn(xframeAlt, RooFit.Name("modelAlt4"), ROOT.RooFit.LineColor(ROOT.kGreen))
                    
                    chi2s_alt1 = xframeAlt.chiSquare("modelAlt1", "obsData")
                    chi2s_alt2 = xframeAlt.chiSquare("modelAlt2", "obsData")
                    chi2s_alt3 = xframeAlt.chiSquare("modelAlt3", "obsData")
                    chi2s_alt4 = xframeAlt.chiSquare("modelAlt4", "obsData")
                    
                    xframeAlt.SetMinimum(0.002)
                    xframeAlt.Draw()
                    
                    c2.SetLogy()
                    c2.SaveAs(carddir + "/plots/TestAfterFit_" + ch + "_Alt.pdf")
            
            # *******************************************************#
            #                                                       #
            #                         Fisher                        #
            #                                                       #
            # *******************************************************#
            
            ofile = open(carddir + "/Fisher/FisherTest_%s.txt" % (ch), "w")
            report = "Results from Fisher Test for category %s \n" % (ch)
            report += "RSS1: %d \n" % (RSS[1]['rss'])
            report += "RSS2: %d \n" % (RSS[2]['rss'])
            report += "RSS3: %d \n" % (RSS[3]['rss'])
            report += "RSS4: %d \n" % (RSS[4]['rss'])
            report += "*******************************************************\n"
            
            order = 0
            print("-" * 25)
            print("function & $\\chi^2$ & RSS & ndof & F-test & result \\\\")
            print("\\hline")
            for o1 in range(1, 4):
                o2 = min(o1 + 1, 5)
                
                CL = fisherTest(RSS[o1]['rss'], RSS[o2]['rss'], o1, o2, RSS[o1]["nbins"])
                
                print("%d par vs %d par CL=%.2f  \n" % (o1, o2, CL), )
                report += "%d par vs %d par: CL=%.2f \n " % (o1, o2, CL)
                if CL > 0.10:  # The function with less parameters is enough
                    if not order:
                        order = o1
                        print("%d par are sufficient" % (o1))
                        # report += "%d par are sufficient \n" % (o1)
                else:
                    print("%d par are needed" % (o2), )
                
                print("\\\\")
            print("\\hline")
            print("-" * 25)
            print("Order is", order, "(" + ch + ")")
            report += "Order is %d (%s)" % (order, ch)
            report += "%d par are sufficient" % (order)
            
            ofile.write(report)
            ofile.close()
            
            if bias:
                print("Running in BIAS mode")
                
                ofile_alt = open(carddir + "/Fisher/FisherTest_alt_%s.txt" % (ch), "w")
                report = "Results from Fisher Test on the alternative function for category %s \n" % (ch)
                report += "RSS1: %d \n" % (RSS_alt[1]['rss'])
                report += "RSS2: %d \n" % (RSS_alt[2]['rss'])
                report += "RSS3: %d \n" % (RSS_alt[3]['rss'])
                report += "RSS4: %d \n" % (RSS_alt[4]['rss'])
                report += "*******************************************************\n"
                
                order_alt = 0
                print("-" * 25)
                print("function & $\\chi^2$ & RSS & ndof & F-test & result \\\\")
                print("\\hline")
                for o1 in range(1, 4):
                    o2 = min(o1 + 1, 5)
                    
                    CL = fisherTest(RSS_alt[o1]['rss'], RSS_alt[o2]['rss'], o1, o2, RSS_alt[o1]["nbins"])
                    
                    print("%d par vs %d par CL=%.2f  \n" % (o1, o2, CL), )
                    report += "%d par vs %d par: CL=%.2f \n " % (o1, o2, CL)
                    if CL > 0.10:  # The function with less parameters is enough
                        if not order_alt:
                            order_alt = o1
                            print("%d par are sufficient" % (o1))
                            # report += "%d par are sufficient \n" % (o1)
                    else:
                        print("%d par are needed" % (o2), )
                    
                    print("\\\\")
                print("\\hline")
                print("-" * 25)
                print("Order is", order_alt, "(" + ch + ")")
                report += "Order is %d (%s)" % (order_alt, ch)
                report += "%d par are sufficient" % (order_alt)
                
                ofile_alt.write(report)
                ofile_alt.close()
            
            if order == 1:
                modelBkg = modelBkg1  # .Clone("Bkg")
                # normzBkg = normzBkg2#.Clone("Bkg_norm")
                fitRes = fitRes1
            elif order == 2:
                modelBkg = modelBkg2  # .Clone("Bkg")
                # normzBkg = normzBkg2#.Clone("Bkg_norm")
                fitRes = fitRes2
            elif order == 3:
                modelBkg = modelBkg3  # .Clone("Bkg")
                # normzBkg = normzBkg3#.Clone("Bkg_norm")
                fitRes = fitRes3
            elif order == 4:
                modelBkg = modelBkg4  # .Clone("Bkg")
                # normzBkg = normzBkg4#.Clone("Bkg_norm")
                fitRes = fitRes4
            else:
                print("Functions with", order + 1, "or more parameters are needed to fit the background")
                exit()
            
            modelBkg.SetName(modelName)
            normBkg.SetName(modelName + "_norm")
            wsfilename = "%s/ws.root" % (WORKDIR)
            wfile_ = ROOT.TFile(wsfilename)
            w_ = wfile_.FindObjectAny("BackgroundWS")
            if not w_.__nonzero__():
                w_ = RooWorkspace("BackgroundWS", "workspace")
            else:
                w_ = wfile_.Get("BackgroundWS")
            print("Storing ", modelName)
            getattr(w_, "import")(modelBkg, RooFit.Rename(modelBkg.GetName()))
            
            getattr(w_, "import")(obsData, RooFit.Rename("data_obs"))
            
            if bias:
                
                if order_alt == 1:
                    modelAlt = modelAlt1  # .Clone("Alt")
                    fitRes = fitRes1_alt
                elif order_alt == 2:
                    modelAlt = modelAlt2  # .Clone("Alt")
                    fitRes = fitRes2_alt
                elif order_alt == 3:
                    modelAlt = modelAlt3  # .Clone("Alt")
                    fitRes = fitRes3_alt
                elif order_alt == 4:
                    modelAlt = modelAlt4  # .Clone("Bkg")
                    fitRes = fitRes4_alt
                else:
                    print("Functions with", order_alt + 1, "or more parameters are needed to fit the background")
                    exit()
                modelAlt.SetName(modelAltName)
                normAlt.SetName(modelAltName + "_norm")
                getattr(w_, "import")(modelAlt, RooFit.Rename(modelAlt.GetName()))
            
            wstatus = w_.writeToFile(wsfilename, False)
        
        wfile = ROOT.TFile("%s/ws.root" % (WORKDIR))
        wBkg = wfile.Get("BackgroundWS")
        print("workspace ", wBkg.Print())
        print("Model name: ", modelName)
        modelBkg = wBkg.pdf(modelName)
        modelAlt = wBkg.pdf(modelAltName)
        modelAlt = wBkg.pdf(modelAltName + "_norm")
        normBkg = wBkg.pdf(modelName + "_norm")
        parSet = modelBkg.getParameters(obsData)
        parSet.Print()
        argList = ROOT.RooArgList(parSet)
        parNames = [argList[i].GetName() for i in range(0, len(parSet))]
        
        # modelExt = RooExtendPdf(modelBkg.GetName(), modelBkg.GetTitle(), modelBkg, normzBkg)
        
        for i in range(0, len(parSet)):
            print(argList[i].GetName(), "   ", argList[i].getVal())
        
        # BIAS STUDY
        # Generate pseudo data
        
        setToys = RooDataSet()
        setToys.SetName("data_toys")
        setToys.SetTitle("Data (toys)")
        
        if not isData:
            print(" - Generating", nBkgEvts, "events for toy data")
            if bias: setToys = modelAlt.generateBinned(RooArgSet(mT), nBkgEvts)
            # fitRes = modelBkg.fitTo(setToys, RooFit.Extended(True), RooFit.Save(1), RooFit.SumW2Error(True), RooFit.Strategy(2), RooFit.Minimizer("Minuit2"), RooFit.PrintLevel(1))
        
        if bias:
            ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
            from ROOT import RooMultiPdf
            pdf_index_string = "pdf_index_%s" % (ch)
            cat = RooCategory(pdf_index_string, "Index of Pdf which is active")
            pdfs = RooArgList(modelBkg, modelAlt)
            roomultipdf = RooMultiPdf("roomultipdf", "All Pdfs", cat, pdfs)
            normulti = RooRealVar("roomultipdf_norm", "Number of background events", nBkgEvts, 0., 1.e6)
        
        # create workspace
        w = RooWorkspace("SVJ", "workspace")
        # Dataset
        # ATT: include isData
        
        # getattr(w, "import")(obsData, RooFit.Rename("Bkg"))
        # if isData: getattr(w, "import")(obsData, RooFit.Rename("data_obs"))
        getattr(w, "import")(obsData, RooFit.Rename("data_obs"))
        # else: getattr(w, "import")(setToys, RooFit.Rename("data_obs"))
        # getattr(w, "import")(obsData, RooFit.Rename("data_obs"))
        getattr(w, "import")(sigData, RooFit.Rename(sig))
        
        if bias:
            getattr(w, "import")(cat, RooFit.Rename(cat.GetName()))
            getattr(w, "import")(normulti, RooFit.Rename(normulti.GetName()))
            getattr(w, "import")(roomultipdf, RooFit.Rename(roomultipdf.GetName()))
        
        for i in range(hist.GetNbinsX()):
            mcstatSysName = "mcstat_%s_%s_bin%d" % (ch, sig, i + 1)
            # print mcstatSysName
            # print sig + "_" + mcstatSysName + "Up"
            mcstatSigUp = getHist(ch, sig + "_" + mcstatSysName + "Up", ifile)
            
            # print "Integral  ", mcstatSigUp.Integral()
            mcstatSigDown = getHist(ch, sig + "_" + mcstatSysName + "Down", ifile)
            mcstatSigHistUp = RooDataHist(sig + "_" + mcstatSysName + "Up", "Data (MC sig)", RooArgList(mT),
                                          mcstatSigUp, 1.)
            mcstatSigHistDown = RooDataHist(sig + "_" + mcstatSysName + "Down", "Data (MC sig)", RooArgList(mT),
                                            mcstatSigDown, 1.)
            getattr(w, "import")(mcstatSigHistUp, RooFit.Rename(sig + "_" + mcstatSysName + "Up"))
            getattr(w, "import")(mcstatSigHistDown, RooFit.Rename(sig + "_" + mcstatSysName + "Down"))
        
        for sysName, sysValue in syst.iteritems():
            if (sysValue[0] == "shape" and "mcstat" not in sysName):
                sysUp = getHist(ch, sig + "_" + sysName + "Up", ifile)
                sysDown = getHist(ch, sig + "_" + sysName + "Down", ifile)
                # print "==> Trigg sys name: ", sig + "_" + sysName + "Down"
                sysSigHistUp = RooDataHist(sig + "_" + sysName + "Up", sysName + " uncertainty", RooArgList(mT), sysUp,
                                           1.)
                sysSigHistDown = RooDataHist(sig + "_" + sysName + "Down", sysName + " uncertainty", RooArgList(mT),
                                             sysDown, 1.)
                getattr(w, "import")(sysSigHistUp, RooFit.Rename(sig + "_" + sysName + "Up"))
                getattr(w, "import")(sysSigHistDown, RooFit.Rename(sig + "_" + sysName + "Down"))
        
        # else: getattr(w, "import")(setToys, RooFit.Rename("data_obs"))
        getattr(w, "import")(modelBkg, RooFit.Rename(modelBkg.GetName()))
        if bias: getattr(w, "import")(modelAlt, RooFit.Rename(modelAlt.GetName()))
        # getattr(w, "import")(normzBkg, RooFit.Rename(normzBkg.GetName()))
        wstatus = w.writeToFile("%sws_%s_%s_%s.root" % (carddir, sig, ch, mode), True)
        
        if wstatus == False:
            print("Workspace", "%sws_%s_%s_%s.root" % (carddir, sig, ch, mode), "saved successfully")
        else:
            print("Workspace", "%sws_%s_%s_%s.root" % (carddir, sig, ch, mode), "not saved successfully")
        workfile = "./ws_%s_%s_%s.root" % (sig, ch, mode)
        # ======   END MODEL GENERATION   ======
        if (verbose): w.Print()
    
    rates = {}
    procLine = ""
    procNumbLine = ""
    rateLine = ""
    binString = ""
    
    if (mode == "template"):
        
        processes.append("Bkg")
        processes[:-1] = []
        rates["Bkg"] = nBkgEvts
        procLine += ("%-43s") % (modelBkg.GetName())
        rateLine += ("%-43s") % (rates["Bkg"])
        binString += (("%-43s") % (ch)) * (2)
        procNumbLine = 1
    
    else:
        i = 1
        bkgrate = 0
        for p in processes:
            rates[p] = getRate(ch, p, ifile)
            bkgrate = rates[p]
            procNumbLine += ("%-43s") % (i)
            procLine += ("%-43s") % (p)
            rateLine += ("%-43.1f") % (bkgrate)
            i += 1
        binString += (("%-43s") % (ch)) * (len(processes) + 1)
    
    rates["data_obs"] = getRate(ch, "data_obs", ifile)
    # print("TEST: ", rates["data_obs"], nBkgEvts)
    rates[sig] = getRate(ch, sig, ifile)
    
    card = "imax 1 number of channels \n"
    card += "jmax * number of backgrounds \n"
    card += "kmax * number of nuisance parameters\n"
    card += "-----------------------------------------------------------------------------------\n"
    
    if (mode == "template"):
        #              card += "shapes   %s  %s    %s    %s    %s\n" % (sig, ch, ifilename, "$CHANNEL/$PROCESS", "$CHANNEL/$PROCESS_SYSTEMATIC")
        #              card += "shapes            %-15s  %-5s    %s%s.root    %s\n" % (sig, ch, WORKDIR, ch, "SVJ:$PROCESS")
        card += "shapes   %s  %s    %s    %s\n" % (modelBkg.GetName(), ch, workfile, "SVJ:$PROCESS")
        card += "shapes   %s  %s    %s    %s    %s\n" % (sig, ch, workfile, "SVJ:$PROCESS", "SVJ:$PROCESS_$SYSTEMATIC")
        card += "shapes   %s  %s    %s    %s\n" % ("data_obs", ch, workfile, "SVJ:$PROCESS")
    
    else:
        card += "shapes   *      *   %s    %s    %s\n" % (
        hist_filename, "$CHANNEL/$PROCESS", "$CHANNEL/$PROCESS_$SYSTEMATIC")
        card += "shapes   data_obs      *   %s    %s\n" % (hist_filename, "$CHANNEL/$PROCESS")
    card += "-----------------------------------------------------------------------------------\n"
    card += "bin               %s\n" % ch
    print("===> Observed data: ", rates["data_obs"])
    card += "observation       %.6f\n" % (rates["data_obs"])
    card += "-----------------------------------------------------------------------------------\n"
    card += "bin                                     %-43s\n" % (binString)
    card += "process                                 %-43s%-43s\n" % (sig, procLine)  # "roomultipdf"
    card += "process                                 %-43s%-43s\n" % ("0", procNumbLine)
    card += "rate                                    %-43.6f%-43s\n" % (
    rates[sig], rateLine)  # signalYield[m].getVal(), nevents
    card += "-----------------------------------------------------------------------------------\n"
    
    for sysName, sysValue in iter(syst.items()):
        
        if (sysValue[0] == "lnN"):
            card += "%-20s%-20s" % (sysName, sysValue[0])
            if (sysValue[1] == "all" and len(sysValue) > 2):
                if (mode == "template"):
                    card += "%-20s" % (sysValue[2]) * (2)
                else:
                    card += "%-20s" % (sysValue[2]) * (len(processes) + 1)
            
            else:
                if (sysValue[1] == "all"):
                    sysValue[1] = copy.deepcopy(processes)
                    sysValue[1].append(sig)
                
                hsysName = "_" + sysName
                hsysNameUp = "_" + sysName + "UP"
                hsysNameDown = "_" + sysName + "DOWN"
                # print "Applying syst on ", sysValue[1]
                if ("sig" in sysValue[1]):
                    if (getRate(ch, sig, ifile) != 0.):
                        sigSys = abs((getRate(ch, sig + hsysNameUp, ifile) - getRate(ch, sig + hsysNameDown, ifile)) / (
                                    2 * getRate(ch, sig, ifile)))
                    else:
                        sigSys = 1
                    if (sigSys < 1. and sigSys > 0.): sigSys = sigSys + 1
                    card += "%-20s" % (sigSys)
                else:
                    card += "%-20s" % ("-")
                for p in processes:
                    if (p in sysValue[1]):
                        if (getRate(ch, p, ifile) != 0.):
                            bkgSys = abs((getRate(ch, p + hsysNameUp, ifile) - getRate(ch, p + hsysNameDown, ifile)) / (
                                        2 * getRate(ch, p, ifile)))
                        else:
                            bkgSys = 1
                        if (bkgSys < 1. and bkgSys > 0.): bkgSys = bkgSys + 1
                        card += "%-20s" % (bkgSys)
                    else:
                        card += "%-20s" % ("-")
        
        
        
        elif (sysValue[0] == "lnU"):
            card += "%-20s%-20s%-20s%-20s" % (sysName, sysValue[0], "-", sysValue[2])
        elif (sysValue[0] == "shape"):
            if ("mcstat" not in sysName):
                card += "%-20s     shape     " % (sysName)
                if ("sig" in sysValue[1]):
                    card += "%-20s" % ("1")
                else:
                    card += "%-20s" % ("-")
                for p in processes:
                    if (p in sysValue[1]):
                        card += "%-20s" % ("1")
                    else:
                        card += "%-20s" % ("-")
            
            
            else:
                # CAMBIARE NOME DELLA SYST
                for samp in sysValue[1]:
                    sampName = ""
                    line = ""
                    if (samp == "sig" or samp == "Sig"):
                        
                        line = "%-20s" % ("1")
                        line += "%-20s" % ("-") * (len(processes))
                        sampName = sig
                    elif (mode != "template"):
                        
                        line = "%-20s" % ("-")
                        lineProc = ["%-20s" % ("-") for x in range(len(processes))]
                        if samp in processes:
                            index = processes.index(samp)
                            lineProc[index] = "1"
                        
                        lineProc = "         ".join(lineProc)
                        line += lineProc
                        sampName = samp
                    
                    else:
                        continue
                    
                    for i in range(hist.GetNbinsX()):
                        sysName = "mcstat_%s_%s_bin%d      " % (ch, sampName, i + 1)
                        card += "%-20s   shape   " % (sysName)
                        card += line
                        card += "\n"
        
        card += "\n"
    
    for par in parNames: card += "%-20s%-20s\n" % (par, "flatParam")
    
    # card += "SF            extArg     1 [0.75,1.25]\n"
    # card += "SF param 1 0.25\n"
    # for par, formula in rateParams.iteritems():
    #       formula = formula.replace("%s", '%.3f' % (eff))
    #       card += "%-20s %-20s %-20s %-20s %s %-20s\n" % (par + "_rate", "rateParam", ch, sig, formula, "SF")
    
    # card += "\n"
    
    outname = "%s%s_%s_%s.txt" % (carddir, sig, ch, mode)
    cardfile = open(outname, 'w')
    cardfile.write(card)
    cardfile.close()
    
    if bias:
        card = card.replace(modelBkg.GetName(), "roomultipdf")
        card.replace("rate                                    %-20.6f%-20.6f\n" % (1, 1),
                     "rate                                    %-20.6f%-20.6f\n" % (10, 1))
        card += "%-35s     discrete\n" % (pdf_index_string)
        outname = "%s%s_%s_%s_bias.txt" % (carddir, sig, ch, mode)
        cardfile = open(outname, 'w')
        cardfile.write(card)
        cardfile.close()
    
    # print card
    return card
