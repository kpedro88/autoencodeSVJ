
string inputFileName = "../training/stat_hists.root";

double minMt = 1000, maxMt = 5000;
int rebin = 5;

vector<string> masses = { "1500", "2000", "2500", "3000", "3500", "4000" };
vector<string> rInvs = { "015", "03", "045", "06", "075" };
vector<string> categories = { "SVJ0_2018", "SVJ1_2018", "SVJ2_2018" };

THStack* prepareStack(TH1D *_background, TH1D *_signal, bool normalize = false)
{
  TH1D *background = (TH1D*)_background->Clone();
  TH1D *signal = (TH1D*)_signal->Clone();
  
  if(normalize){
    background->Scale(1./background->GetEntries());
    signal->Scale(1./signal->GetEntries());
  }
  
  THStack *stack = new THStack("stack", "");
  
  background->Rebin(rebin);
  background->Scale(1./rebin);
  background->SetLineColor(kGreen+2);
  background->Sumw2(false);
  background->GetXaxis()->SetTitle("m_{t} (GeV)");
  
  stack->Add(background);
  
  signal->Rebin(rebin);
  signal->Scale(1./rebin);
  signal->SetLineColor(kRed);
  signal->Sumw2(false);
  
  stack->Add(signal);
  
  stack->Draw();
  stack->GetXaxis()->SetTitle("m_{t} (GeV)");
  stack->GetXaxis()->SetRangeUser(minMt, maxMt);
  
  return stack;
}

void drawMtPlots()
{
  TFile *inFile = TFile::Open(inputFileName.c_str());
  
  TCanvas *canvas = new TCanvas("canvas", "canvas", 1280, 800);
  canvas->Divide(2, 2);
  
  TCanvas *canvasNormed = new TCanvas("canvas normalized", "canvas normalized", 1280, 800);
  canvasNormed->Divide(2, 2);
  
  int iPad = 1;
  for(string category : categories){
    inFile->cd(category.c_str());
    
    TH1D *QCDhist = (TH1D*)inFile->Get((category+"/QCD").c_str());
    TH1D *backgroundHist = (TH1D*)inFile->Get((category+"/Bkg").c_str());
    TH1D *dataHist = (TH1D*)inFile->Get((category+"/data_obs").c_str());
    
    string mass = masses[0];
    string rInv = rInvs[0];
    
    string signalHistName = "SVJ_mZprime"+mass+"_mDark20_rinv"+rInv+"_alphapeak";
    TH1D *signalHist = (TH1D*)inFile->Get((category+"/"+signalHistName).c_str());
    
    
    canvas->cd(iPad);
    
    THStack *stack = prepareStack(QCDhist, signalHist);
    stack->SetTitle(category.c_str());
    stack->Draw();
    
    canvasNormed->cd(iPad);
    
    THStack *stackNormed = prepareStack(QCDhist, signalHist, true);
    
    stackNormed->SetTitle(category.c_str());
    stackNormed->GetYaxis()->SetRangeUser(0., 0.4);
    stackNormed->Draw("nostack");
    
    iPad++;
  }
  
  canvas->Update();
  
}
