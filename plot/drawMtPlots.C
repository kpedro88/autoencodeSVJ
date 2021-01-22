
string inputFileName = "../training/stat_hists.root";

double minMt = 1000, maxMt = 5000;
int rebin = 5;

vector<string> masses = { "1500", "2000", "2500", "3000", "3500", "4000" };
map<string, int> massColors = {
  { "1500", kViolet+2 },
  { "2000", kBlue },
  { "2500", kCyan },
  { "3000", kGreen },
  { "3500", kOrange },
  { "4000", kRed },
};

/*
map<string, double> svjCrossSections = { // fb (?)
  {"1500", 0.77},
  {"2000", 0.1849},
  {"2500", 0.04977},
  {"3000", 0.0155},
  {"3500", 0.005036},
  {"4000", 0.001688},
  
//  {"1500", 2.658e-10 * 1e12 }, // mb -> fb
//  {"2000", 5.150e-11 * 1e15 }, // b -> fb
//
//  {"2500", 5.150e-11 * 1e15 }, // b -> fb
//  {"3000", 5.150e-11 * 1e15 }, // b -> fb
//  {"3500", 5.150e-11 * 1e15 }, // b -> fb
//  {"4000", 5.150e-11 * 1e15 }, // b -> fb
};
*/
 
// from Maurizio
map<string, double> svjCrossSections = { // mb
  {"1500" , 2.631e-10 * 1e12 }, // mb -> fb
//  {"1500GeV_0.15" , 2.631e-10 * 1e12 }, // mb -> fb
//  {"1500GeV_0.30" , 2.615e-10 * 1e12 }, // mb -> fb
//  {"1500GeV_0.45" , 2.692e-10 * 1e12 }, // mb -> fb
//  {"1500GeV_0.60" , 2.567e-10 * 1e12 }, // mb -> fb
//  {"1500GeV_0.75" , 2.527e-10 * 1e12 }, // mb -> fb
  {"2000" , 5.168e-11 * 1e12 }, // mb -> fb
//  {"2000GeV_0.15" , 5.168e-11 * 1e12 }, // mb -> fb
//  {"2000GeV_0.30" , 5.237e-11 * 1e12 }, // mb -> fb
//  {"2000GeV_0.45" , 5.124e-11 * 1e12 }, // mb -> fb
//  {"2000GeV_0.60" , 5.112e-11 * 1e12 }, // mb -> fb
//  {"2000GeV_0.75" , 5.119e-11 * 1e12 }, // mb -> fb
  {"2500GeV" , 1.366e-11 * 1e12 }, // mb -> fb
//  {"2500GeV_0.15" , 1.366e-11 * 1e12 }, // mb -> fb
//  {"2500GeV_0.30" , 1.345e-11 * 1e12 }, // mb -> fb
//  {"2500GeV_0.45" , 1.397e-11 * 1e12 }, // mb -> fb
//  {"2500GeV_0.60" , 1.415e-11 * 1e12 }, // mb -> fb
//  {"2500GeV_0.75" , 1.396e-11 * 1e12 }, // mb -> fb
  {"3000" , 4.544e-12 * 1e12 }, // mb -> fb
//  {"3000GeV_0.15" , 4.544e-12 * 1e12 }, // mb -> fb
//  {"3000GeV_0.30" , 4.620e-12 * 1e12 }, // mb -> fb
//  {"3000GeV_0.45" , 4.468e-12 * 1e12 }, // mb -> fb
//  {"3000GeV_0.60" , 4.562e-12 * 1e12 }, // mb -> fb
//  {"3000GeV_0.75" , 4.572e-12 * 1e12 }, // mb -> fb
  {"3500" , 1.816e-12 * 1e12 }, // mb -> fb
//  {"3500GeV_0.15" , 1.816e-12 * 1e12 }, // mb -> fb
//  {"3500GeV_0.30" , 1.855e-12 * 1e12 }, // mb -> fb
//  {"3500GeV_0.45" , 1.790e-12 * 1e12 }, // mb -> fb
//  {"3500GeV_0.60" , 1.822e-12 * 1e12 }, // mb -> fb
//  {"3500GeV_0.75" , 1.845e-12 * 1e12 }, // mb -> fb
  {"4000" , 8.700e-13 * 1e12 }, // mb -> fb
//  {"4000GeV_0.15" , 8.700e-13 * 1e12 }, // mb -> fb
//  {"4000GeV_0.30" , 8.658e-13 * 1e12 }, // mb -> fb
//  {"4000GeV_0.45" , 8.558e-13 * 1e12 }, // mb -> fb
//  {"4000GeV_0.60" , 8.560e-13 * 1e12 }, // mb -> fb
//  {"4000GeV_0.75" , 8.698e-13 * 1e12 }, // mb -> fb
};


double qcdNgenEvents = 3500000;
//double qcdCrossSection = 3358729;
//double qcdCrossSection = 1918063.6;

// from Maurizio
double qcdCrossSection = 1.015e-08 * 1e12; // mb -> fb

//double qcdCrossSection = 96 * 1e12; // mb -> fb

double lumi2018 = 59.8; // fb^-1

vector<string> rInvs = { "015", "03", "045", "06", "075" };
vector<string> rInvTitles = { "0.15", "0.30", "0.45", "0.60", "0.75" };

vector<string> categories = { "SVJ0_2018", "SVJ1_2018", "SVJ2_2018" };
vector<string> categoryTitles = { "0 SV Jets", "1 SV Jet", "2 SV Jets" };

enum ENormType {
  kNoNormalization,
  kNormToOne,
  kNormToCrossSection
};

vector<ENormType> normTypes = {kNoNormalization, kNormToOne, kNormToCrossSection};
map<ENormType, string> normTypeTitles = {
  {kNoNormalization , "no_normalization"},
  {kNormToOne , "normed_to_one"},
  {kNormToCrossSection , "normed_to_xsec"},
};

tuple<map<string, THStack*>, TLegend*> prepareStack(TH1D *_background, map<string, TH1D*> _signals, ENormType normalize)
{
  TH1D *background = (TH1D*)_background->Clone();
  map<string, TH1D*> signals;
  for(auto &[key, hist] : _signals) signals[key] = (TH1D*)hist->Clone();
  
  if(normalize == kNormToOne){
    background->Scale(1./background->GetEntries());
    for(auto &[key, hist] : signals) hist->Scale(1./hist->GetEntries());
  }
  else if(normalize == kNormToCrossSection){
    
//    cout<<"scale: "<<qcdCrossSection*lumi2018<<endl;
//    cout<<"scale2: "<<background->GetEntries()/qcdNgenEvents<<endl;
    
    background->Scale(qcdCrossSection * lumi2018 * background->GetEntries()/qcdNgenEvents);
    for(auto &[key, hist] : signals) hist->Scale(svjCrossSections[key] * lumi2018 * hist->GetEntries()/100000);
  }
  
  map<string, THStack*> stacks;
  TLegend *legend = new TLegend(0.5, 0.4, 0.9, 0.9);
  
  background->Rebin(rebin);
  background->Scale(1./rebin);
  background->SetLineColor(kBlack);
  if(normalize != kNormToOne) background->SetFillColorAlpha(kBlack, 0.3);
  background->Sumw2(false);
  background->GetXaxis()->SetTitle("m_{t} (GeV)");
  
  legend->AddEntry(background, "QCD", "lf");
  
  for(auto &[key, hist] : signals){
    
    stacks[key] = new THStack(("stack_"+key).c_str(), "");
    stacks[key]->Add(background);
    
    hist->Rebin(rebin);
    hist->Scale(1./rebin);
    hist->SetLineColor(massColors[key]);
    hist->Sumw2(false);
    
    stacks[key]->Add(hist);
    legend->AddEntry(hist, ("m = "+key+" GeV").c_str(), "l");
    
    stacks[key]->Draw();
    stacks[key]->GetXaxis()->SetTitle("m_{t} (GeV)");
    stacks[key]->GetXaxis()->SetRangeUser(minMt, maxMt);
    stacks[key]->SetMinimum(normalize==kNormToOne ? 0.0 : (normalize==kNormToCrossSection ? 1e2 : 0));
    stacks[key]->SetMaximum(normalize==kNormToOne ? 0.1 : (normalize==kNormToCrossSection ? 1e8 : 300));
  }
  
  return {stacks, legend};
}

void drawMtPlots()
{
  gStyle->SetLineScalePS(1);
  
  TFile *inFile = TFile::Open(inputFileName.c_str());
  
  map<ENormType, TCanvas*> canvases;
  
  for(ENormType normType : normTypes){
    string title = "canvas_"+normTypeTitles[normType];
    canvases[normType] = new TCanvas(title.c_str(), title.c_str(), 2880, 1800);
    canvases[normType]->Divide(5, 3);
  }
  
  for(int iCategory=0; iCategory<categories.size(); iCategory++){
    string category = categories[iCategory];
    
    inFile->cd(category.c_str());
    TH1D *QCDhist = (TH1D*)inFile->Get((category+"/QCD").c_str());
    TH1D *backgroundHist = (TH1D*)inFile->Get((category+"/Bkg").c_str());
    TH1D *dataHist = (TH1D*)inFile->Get((category+"/data_obs").c_str());
    
    for(int iRinv=0; iRinv<rInvs.size(); iRinv++){
      string rInv = rInvs[iRinv];
      string padTitle = categoryTitles[iCategory] + ", r_{inv} = " + rInvTitles[iRinv];
      
      map<string, TH1D*> signalHists;
      
      for(string mass : masses){
        string signalHistName = "SVJ_mZprime"+mass+"_mDark20_rinv"+rInv+"_alphapeak";
        signalHists[mass] = (TH1D*)inFile->Get((category+"/"+signalHistName).c_str());
      }
      
      int iPad = iCategory*rInvs.size() + iRinv + 1;
      
      for(ENormType normType : normTypes){
        canvases[normType]->cd(iPad);
        auto [stacks, legend] = prepareStack(QCDhist, signalHists, normType);
        
        bool first = true;
        for(auto &[key, stack] : stacks){
          stack->SetTitle(padTitle.c_str());
          string drawOptions = (normType==kNormToOne) ? "nostack" : "";
          if(!first) drawOptions += "same";
          stack->Draw(drawOptions.c_str());
          first = false;
        }
        legend->Draw();
        
        if(normType == kNormToCrossSection) gPad->SetLogy();
        
      }
    }
  }
  
  
  for(ENormType normType : normTypes){
    string outFileName = "mt_distributions_"+normTypeTitles[normType]+".pdf";
    canvases[normType]->SaveAs(outFileName.c_str());
  }
}

The process proceeds though a loop of charged SM particles. It could also go through a loop containing new charged particles (SUSY) or through the s-channel with spin-even resonances (axions, monopoles).


