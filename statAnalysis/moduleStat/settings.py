import collections

#*********************************
#                                *
#       List of channels         *
#                                *
#*********************************


### List of histos to include in the root files
histos = {"SVJ0":"h_Mt_SVJ0",
          "SVJ1" :"h_Mt_SVJ1",
          "SVJ2": "h_Mt_SVJ2",
          }

#histos = {"BDT0":"h_Mt_BDT0","BDT1" :"h_Mt_BDT1", "BDT2": "h_Mt_BDT2", "CRBDT0":"h_Mt_CRBDT0", "CRBDT1":"h_Mt_CRBDT1", "CRBDT2":"h_Mt_CRBDT2"}

#histos = {"BDT0":"h_Mt"}

### List of regions for which creating the datacards
channels = [ "SVJ0", "SVJ1", "SVJ2" ]

#channels = [ "BDT0"]

#*********************************
#                                *
#       List of systematics      *
#                                *
#*********************************

syst = collections.OrderedDict()
syst["lumi"] = ("lnN", "all", 1.10)
syst["trigger"] = ("lnN", "all", 1.02)
syst["BkgRate"] = ("lnU", "Bkg", 4.)
#syst["mcstat"] = ("shape", ("QCD", "TT", "WJets", "ZJets", "sig"))
syst["mcstat"] = ("shape", ["sig"])
#syst["trig"] = ("shape", ["sig"])
#syst["JER"] = ("shape", ["sig"])
#syst["JEC"] = ("shape", ["sig"])

#****************************************
#                                       *
#       List of rateParam funtions      *
#                                       *
#****************************************

rateParams = {}
rateParams["SVJ0_2018"] = "TMath::Power(TMath::Range(0.01,0.99,@0),1)*TMath::Power(1-TMath::Range(0.01,0.99,@0*%s),1)/(TMath::Power(1-%s,1))"
rateParams["SVJ1_2018"] = "TMath::Power(TMath::Range(0.01,0.99,@0),2)*TMath::Power(1-TMath::Range(0.01,0.99,@0*%s),0)/(TMath::Power(1-%s,0))"
rateParams["SVJ2_2018"] = "TMath::Power(TMath::Range(0.01,0.99,@0),1)*TMath::Power(1-TMath::Range(0.01,0.99,@0*%s),1)/(TMath::Power(1-%s,1))"


#*********************************
#                                *
#       List of backgrounds      *
#                                *
#*********************************

processes = ["QCD"]

#*********************************
#                                *
#         List of signals        *
#                                *
#*********************************

vec1 = ("500", "20", "03", "peak")
vec2 = ("600", "20", "03", "peak")
vec3 = ("700", "20", "03", "peak")
vec4 = ("800", "20", "03", "peak")
vec5 = ("900", "20", "03", "peak")
vec6 = ("1000", "20", "03", "peak")
vec7 = ("1100", "20", "03", "peak")
vec8 = ("1200", "20", "03", "peak")
vec9 = ("1300", "20", "03", "peak")
vec10 = ("1400", "20", "03", "peak")
vec11 = ("1500", "20", "03", "peak")
vec12 = ("1600", "20", "03", "peak")
vec13 = ("1700", "20", "03", "peak")
vec14 = ("1800", "20", "03", "peak")
vec15 = ("1900", "20", "03", "peak")
vec16 = ("2000", "20", "03", "peak")
vec17 = ("2100", "20", "03", "peak")
vec18 = ("2200", "20", "03", "peak")
vec19 = ("2300", "20", "03", "peak")
vec20 = ("2400", "20", "03", "peak")
vec21 = ("2500", "20", "03", "peak")
vec22 = ("2600", "20", "03", "peak")
vec23 = ("2700", "20", "03", "peak")
vec24 = ("2800", "20", "03", "peak")
vec25 = ("2900", "20", "03", "peak")
vec26 = ("3000", "20", "03", "peak")
vec27 = ("3100", "20", "03", "peak")
vec28 = ("3200", "20", "03", "peak")
vec29 = ("3300", "20", "03", "peak")
vec30 = ("3400", "20", "03", "peak")
vec31 = ("3500", "20", "03", "peak")
vec32 = ("3600", "20", "03", "peak")
vec33 = ("3700", "20", "03", "peak")
vec34 = ("3800", "20", "03", "peak")
vec35 = ("3900", "20", "03", "peak")
vec36 = ("4000", "20", "03", "peak")
vec37 = ("4100", "20", "03", "peak")
vec38 = ("4200", "20", "03", "peak")
vec39 = ("4300", "20", "03", "peak")
vec40 = ("4400", "20", "03", "peak")
#vec41 = ("4500", "20", "03", "peak")
vec41 = ("3000", "20", "0", "peak")

#vec1 = ("2000", "20", "03", "peak")


#sigpoints = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15, vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23, vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31, vec32, vec33, vec34, vec35, vec36, vec37, vec38, vec39, vec40]

# sigpoints = [ vec11, vec12, vec13, vec14, vec15, vec16, vec17, vec18, vec19, vec20, vec21, vec22, vec23, vec24, vec25, vec26, vec27, vec28, vec29, vec30, vec31, vec32, vec33, vec34, vec35, vec36, vec37, vec38, vec39, vec40]

sigpoints = [ vec16 ]




#sigpoints = [ vec11, vec12, vec13, vec14, vec15]
#sigpoints = [ vec11, vec12]



#sigpoints = [vec11]


#sigpoints = [s for s in sigpoints if sigpoints.index(s) % 2 == 0 ]
