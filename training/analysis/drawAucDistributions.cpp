#include "Helpers.hpp"

#include "Result.hpp"
#include "ResultsProcessor.hpp"

string aucsPath =  "../trainingResults/aucs/standardScaler/";
string resultsPath =  "../trainingResults/trainingRuns/standardScaler/";
string filePattern = "hlf_eflow3_8_v";
string plotsTitle = "StandardScaler";

vector<double> r_invs = {0.15, 0.30, 0.45, 0.60, 0.75};
vector<double> masses = {1500, 2000, 2500, 3000, 3500, 4000};
vector<int> colors = {kRed, kOrange, kGreen, kGreen+2, kBlue, kViolet, kBlack};

const double xMin = 0.6;
const double xMax = 1.0;

const int nBinsRinv = 130;
const int nBinsMass = 50;

TH1D* getHistogramForVariable(const vector<ModelStats> &stats, double variableValue, bool forMass)
{
  string title = forMass ? "mass" : "r_inv";
  title += " hist" + to_string(variableValue);
  TH1D *hist = new TH1D(title.c_str(), title.c_str(), forMass ? nBinsMass : nBinsRinv, xMin, xMax);
  
  for(ModelStats stat : stats){
    
    for(Result result : stat.results){
      if((forMass ? result.mass : result.r_inv) == variableValue){
        hist->Fill(result.AUC);
      }
    }
  }
  return hist;
}

void drawHistsForVariable(const vector<ModelStats> &stats, bool forMass)
{
  TLegend *leg = new TLegend(0.1, 0.6, 0.5, 0.9);
  
  cout<<"AUCs per "<<(forMass ? "mass" : "r_inv")<<": "<<endl;
  cout<<"mean\tmeanErr\twidth\twidthErr\tmax\tmaxErr"<<endl;
  
  for(int i=0; i<(forMass ? masses.size() : r_invs.size()); i++){
    
    TH1D *hist = getHistogramForVariable(stats, forMass ? masses[i] : r_invs[i], forMass);
    
    if(i==0){
      hist->SetTitle(plotsTitle.c_str());
      hist->GetXaxis()->SetTitle("AUC");
      hist->GetYaxis()->SetTitle("# trainings");
    }
    
    hist->Sumw2();
    hist->SetLineColor(colors[i]);
    string title = forMass ? "m = " : "r_{inv} = ";
    title += to_string_with_precision(forMass ? masses[i] : r_invs[i], forMass ? 0 : 2);
    if(forMass) title += " GeV";
    leg->AddEntry(hist, title.c_str(), "l");
    
    hist->Draw(i==0 ? "" : "same");
    
    cout<<(forMass ? masses[i] : r_invs[i])<<"\t";
    
    
    cout<<hist->GetMean()<<"\t"<<hist->GetMeanError()<<"\t";
    cout<<hist->GetStdDev()<<"\t"<<hist->GetStdDevError()<<"\t";
    cout<<hist->GetXaxis()->GetBinCenter(hist->FindLastBinAbove(0))<<"\t"<<hist->GetXaxis()->GetBinWidth(0)/2.<<endl;
    
  }
  
  leg->Draw();
}

int main()
{
  cout<<"Starting drawAucDistributions"<<endl;
  gStyle->SetOptStat(0);
  useCommaAsDecimalSeparator();
 
  cout<<"Creating application"<<endl;
  TApplication app("", 0, {});
  
  cout<<"Reading results from files"<<endl;
  auto resultsProcessor = make_unique<ResultsProcessor>();
  vector<ModelStats> stats = resultsProcessor->getModelStatsFromPathMarchingPatter(aucsPath, resultsPath, filePattern);
  
  cout<<"Plotting results"<<endl;
  TCanvas *canvas = new TCanvas("c1", "c1", 1000, 2000);
  canvas->Divide(1, 2);
  
  canvas->cd(1);
  drawHistsForVariable(stats, false);
    
  canvas->cd(2);
  drawHistsForVariable(stats, true);
  
  canvas->Update();
  
  cout<<"Running the application"<<endl;
  app.Run();
  return 0;
}
