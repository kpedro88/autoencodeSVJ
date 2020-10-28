#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

string basePath =  "../trainingResults/aucs/";
string filePattern = "hlf_eflow3_8_v";

vector<double> r_invs = {0.15, 0.30, 0.45, 0.60, 0.75};
vector<double> masses = {1500, 2000, 2500, 3000, 3500, 4000};
vector<int> colors = {kRed, kOrange, kGreen, kGreen+2, kBlue, kViolet, kBlack};

const double xMin = 0.6;
const double xMax = 0.9;

const int nBinsMass = 50;
const int nBinsRinv = 100;


struct Result {
  string name;
  double mass, r_inv, AUC;
  
  void print(){
    cout<<name<<"\t"<<mass<<"\t"<<r_inv<<"\t"<<AUC<<endl;
  }
};

template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
  ostringstream out;
  out.precision(n);
  out << fixed << a_value;
  return out.str();
}

vector<string> splitByComma(string input)
{
  string delimiter = ",";
  size_t pos = 0;
  
  vector<string> results;
  
  while((pos = input.find(delimiter)) != string::npos){
    results.push_back(input.substr(0, pos));
    input.erase(0, pos + delimiter.length());
  }
  results.push_back(input.substr(0, pos));
  
  
  return results;
}

vector<Result> getResultsFromFile(string inFilePath)
{
  ifstream infile(inFilePath);
  
  vector<Result> results;
  
  string line;
  
  bool firstLine = true;
  
  while(getline(infile, line)){
    if(firstLine){
      firstLine = false;
      continue;
    }
    vector<string> lineByCommas = splitByComma(line);
    
    if(lineByCommas.size() != 5) continue;
    
    Result r;
    
    r.name = lineByCommas[1];
    r.AUC = stod(lineByCommas[2]);
    r.mass = stod(lineByCommas[3]);
    r.r_inv = stod(lineByCommas[4]);
    
    results.push_back(r);
  }
  return results;
}

vector<string> getFileInPathMatchingPattern(string path, string pattern)
{
  DIR* dirp = opendir(path.c_str());
  struct dirent * dp;

  vector<string> filePaths;
  
  while((dp = readdir(dirp))){
    string fileName = dp->d_name;
    
    if(fileName.find(pattern) != string::npos){
      filePaths.push_back(path+fileName);
    }
  }
  closedir(dirp);
  
  return filePaths;
}

TH1D* getHistogramForMass(const vector<Result> &results, double mass)
{
  string title = "hist"+to_string(mass);
  TH1D *hist = new TH1D(title.c_str(), title.c_str(), nBinsMass, xMin, xMax);
  
  for(Result result : results){
    if(result.mass == mass){
      hist->Fill(result.AUC);
    }
  }
  return hist;
}

TH1D* getHistogramForRinv(const vector<Result> &results, double r_inv)
{
  string title = "hist"+to_string(r_inv);
  TH1D *hist = new TH1D(title.c_str(), title.c_str(), nBinsRinv, xMin, xMax);
  
  for(Result result : results){
    if(result.r_inv == r_inv){
      hist->Fill(result.AUC);
    }
  }
  return hist;
}

vector<Result> getResults()
{
  vector<string> paths = getFileInPathMatchingPattern(basePath, filePattern);
  
  vector<Result> results;
  
  for(string path : paths){
    vector<Result> resultsInFile = getResultsFromFile(path);
    
    for(Result result : resultsInFile){
      results.push_back(result);
    }
  }
  return results;
}

void drawHistsByRinv(const vector<Result> &results)
{
  TLegend *leg = new TLegend(0.1, 0.6, 0.5, 0.9);
  
  for(int i=0; i<r_invs.size(); i++){
    
    TH1D *hist = getHistogramForRinv(results, r_invs[i]);
    
    if(i==0){
      hist->SetTitle("");
      hist->GetXaxis()->SetTitle("AUC");
      hist->GetYaxis()->SetTitle("# trainings");
    }
    
    hist->Sumw2();
    
    hist->SetLineColor(colors[i]);
    
    leg->AddEntry(hist, ("r_{inv} = "+to_string_with_precision(r_invs[i], 2)).c_str(), "l");
    
    hist->Draw(i==0 ? "" : "same");
  }
  
  leg->Draw();
}

void drawHistsByMass(const vector<Result> &results)
{
  TLegend *leg = new TLegend(0.1, 0.6, 0.5, 0.9);
  
  for(int i=0; i<masses.size(); i++){
    
    TH1D *hist = getHistogramForMass(results, masses[i]);
    
    if(i==0){
      hist->SetTitle("");
      hist->GetXaxis()->SetTitle("AUC");
      hist->GetYaxis()->SetTitle("# trainings");
    }
    
    hist->Sumw2();
    hist->SetLineColor(colors[i]);
    
    leg->AddEntry(hist, ("m = "+to_string_with_precision(masses[i], 0)+" GeV").c_str(), "l");
    
    hist->Draw(i==0 ? "" : "same");
  }
  
  leg->Draw();
}

void drawAucDistributions()
{
  gStyle->SetOptStat(0);
  
  vector<Result> results = getResults();
  
  TCanvas *canvas = new TCanvas("c1", "c1", 1000, 2000);
  canvas->Divide(1, 2);
  
  canvas->cd(1);
  drawHistsByRinv(results);
    
  canvas->cd(2);
  drawHistsByMass(results);
}
