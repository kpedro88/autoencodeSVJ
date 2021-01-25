#include "Helpers.hpp"
#include "ParallelTreeChain.h"
#include "LorentzMock.h"

using std::fabs;
using std::chrono::microseconds;  
using std::chrono::duration_cast;
using std::string;
using std::endl;
using std::cout;
using std::vector;
using std::pair; 
using std::to_string;
using std::stringstream; 
using std::setw;

namespace vectorTypes{
enum vectorType {
  Lorentz,
  Mock,
  Map
};
};

namespace Cuts {
enum CutType {
  leptonCounts,
  jetCounts,
  jetEtas,
  jetDeltaEtas,
  metRatio,
  jetPt,
  jetDiJet,
  metValue,
  metRatioTight,
  selection,
  COUNT
};

std::map<CutType, string> CutName {
  {leptonCounts, "0 Passing Leptons"},
  {jetCounts, "n Jets > 1"},
  {jetEtas, "abs jet Etas < 2.4"},
  {jetDeltaEtas, "abs DeltaEta < 1.5"},
  {metRatio,"MET/M_T > 0.15"},
  {jetPt, "Jet PT > 200"},
  {jetDiJet, "Dijet veto"},
  {metValue, "M_T > 1500"},
  {metRatioTight, "MET/M_T > 0.25"},
  {selection, "final selection"}
};
};

namespace Hists {
enum HistType {
  dEta,
  dPhi,
  tRatio,
  met2,
  mjj,
  metPt,
  
  pre_1pt,
  pre_2pt,
  post_1pt,
  post_2pt,
  
  pre_lep,
  post_lep,
  
  pre_MT,
  pre_mjj,
  
  COUNT
};
}; 

// import for backportability (;-<)
using namespace vectorTypes; 
using namespace Cuts; 

class SVJFinder {
public:
  /// CON/DESTRUCTORS
  ///
  
  // constructor, requires argv as input
  SVJFinder(char **argv) {
    start();
    tStart(programstart);
    log("ROOT");
    log();
    log("-----------------------------------");
    log(":          SVJAnalysis            :");
    log("-----------------------------------");
    log();
    inputspec = argv[1];
    log(string("File list to open: " + inputspec));
    
    sample = argv[2];
    log(string("Sample name: " + sample));
    
    outputdir = argv[3];
    log(string("Output directory: " + outputdir));
    log();
    debug = std::atoi(argv[4]);
    timing = std::atoi(argv[5]);
    saveCuts = std::atoi(argv[6]);
    nMin = std::stoi(argv[7]);
    nMax = std::stoi(argv[8]);
    
    if (nMin < 0) nMin = 0;
    
    log("SVJ object created");
    end();
    logt();
    log();
  }
  
  // destructor for dynamically allocated data
  ~SVJFinder() {
    start();
    
    Debug(true);
    log();
    logp("Quitting; cleaning up class variables...  ");
    
    DelVector(varValues);
    DelVector(vectorVarValues);
    DelVector(LorentzVectors);
    DelVector(MockVectors);
    DelVector(MapVectors);
    DelVector(hists);
    delete chain;
    chain = nullptr;
    file->Close();
    file = nullptr;
    logr("Success");
    end();
    logt();
    log();
    double pdur = tsRaw(tEnd(programstart));
    logp("total program duration: ");
    lograw(pdur);
    logr("s");
    
    log();
    // DelVector(varLeaves);
    // DelVector(vectorVarLeaves);
    // for (vector<TLeaf*> vec : compVectors)
    //     DelVector(vec);
  }
  
  /// FILE HANDLERS
  ///
  
  // sets up tfile collection and returns a pointer to it
  // TFileCollection *MakeFileCollection() {
  //     start();
  //     log("Loading File Collection from " + inputspec);
  //     if (fc)
  //         delete fc;
  //     fc = new TFileCollection(sample.c_str(), sample.c_str(), inputspec.c_str());
  // file = new TFile((outputdir + "/" + sample + "_output.root").c_str(), "RECREATE");
  //     log("Success: Loaded " + std::to_string(fc->GetNFiles())  + " file(s).");
  //     end();
  //     logt();
  //     log();
  //     return fc;
  // }
  
  // sets up paralleltreechain and returns a pointer to it
  ParallelTreeChain* MakeChain() {
    start();
    log("Creating file chain with tree type 'Delphes'...");
    chain = new ParallelTreeChain();
    outputTrees = chain->GetTrees(inputspec, "Delphes");
    
    selectionIndex.resize(outputTrees.size());
    
    file = new TFile((outputdir + "/" + sample + "_output.root").c_str(), "RECREATE");
    nEvents = (Int_t)chain->GetEntries();
    
    if (nMax < 0 || nMax > nEvents) nMax = nEvents;
    
    log("Success");
    end();
    logt();
    log();
    return chain;
  }
  
  /// VARIABLE TRACKER FUNCTIONS
  ///
  
  // creates, assigns, and returns tlorentz vector pointer to be updated on GetEntry
  vector<TLorentzVector>* AddLorentz(string vectorName, vector<string> components) {
    start();
    assert(components.size() == 4);
    AddCompsBase(vectorName, components);
    size_t i = LorentzVectors.size();
    subIndex.push_back(std::make_pair(i, vectorType::Lorentz));
    vector<TLorentzVector>* ret = new vector<TLorentzVector>;
    LorentzVectors.push_back(ret);
    logr("Success");
    end();
    logt();
    return ret;
  }
  
  // creates, assigns, and returns mock tlorentz vector pointer to be updated on GetEntry
  vector<LorentzMock>* AddLorentzMock(string vectorName, vector<string> components) {
    start();
    assert(components.size() > 1 && components.size() < 5);
    AddCompsBase(vectorName, components);
    size_t i = MockVectors.size();
    subIndex.push_back(std::make_pair(i, vectorType::Mock));
    vector<LorentzMock>* ret = new vector<LorentzMock>;
    MockVectors.push_back(ret);
    logr("Success");
    end();
    logt();
    return ret;
  }
  
  // creates, assigns, and returns general double vector pointer to be updated on GetEntry
  vector<vector<double>>* AddComps(string vectorName, vector<string> components) {
    start();
    AddCompsBase(vectorName, components);
    size_t i = MapVectors.size();
    subIndex.push_back(std::make_pair(i, vectorType::Map));
    vector<vector<double>>* ret = new vector<vector<double>>;
    MapVectors.push_back(ret);
    logr("Success");
    end();
    logt();
    return ret;
  }
  
  // creates, assigns, and returns a vectorized single variable pointer to be updates on GetEntry
  vector<double>* AddVectorVar(string vectorVarName, string component) {
    start();
    logp("Adding 1 component to vector var " + vectorVarName + "...  ");
    int i = int(vectorVarValues.size());
    vectorVarIndex[vectorVarName] = i;
    vectorVarLeaves.push_back(chain->FindLeaf(component));
    vector<double>* ret = new vector<double>;
    vectorVarValues.push_back(ret);
    logr("Success");
    end();
    logt();
    // log(vectorVarIndex.size());
    // log(vectorVarValues.back().size());
    // log(i);
    return ret;
  }
  
  // creates, assigns, and returns a singular double variable pointer to update on GetEntry
  double* AddVar(string varName, string component) {
    start();
    logp("Adding 1 component to var " + varName + "...  ");
    size_t i = varLeaves.size();
    varIndex[varName] = i;
    double* ret = new double;
    varLeaves.push_back(chain->FindLeaf(component));
    varValues.push_back(ret);
    logr("Success");
    end();
    logt();
    return ret;
  }
  
  /// ENTRY LOADING
  ///
  
  void reloadLeaves() {
    
  }
  
  // get the ith entry of the TChain
  void GetEntry(int entry = 0) {
    assert(entry < chain->GetEntries());
    logp("Getting entry " + to_string(entry) + "...  ");
    int treeId = chain->GetEntry(entry);
    currentEntry = entry;
    if (chain->currentEntry == 0) {
      bool last = debug;
      Debug(true);
      logp("");
      Debug(last);
      cout << "Processing tree " << chain->currentTree + 1 << " of " << chain->size() << endl;
    }
    for (size_t i = 0; i < subIndex.size(); ++i) {
      switch(subIndex[i].second) {
        case vectorType::Lorentz: {
          SetLorentz(i, subIndex[i].first, treeId);
          break;
        }
        case vectorType::Mock: {
          SetMock(i, subIndex[i].first, treeId);
          break;
        }
        case vectorType::Map: {
          SetMap(i, subIndex[i].first, treeId);
          break;
        }
      }
    }
    
    for (size_t i = 0; i < varValues.size(); ++i) {
      SetVar(i, treeId);
    }
    
    for (size_t i = 0; i < vectorVarValues.size(); ++i) {
      SetVectorVar(i, treeId);
    }
    // cout << vectorVarValues.size() << endl;
    // for (size_t i = 0; i < vectorVarValues.size(); ++i) {
    //     cout << i << " | ";
    //     for (size_t j = 0; j < vectorVarValues[i].size(); ++j) {
    //         cout << j << ": " << vectorVarValues[i][j] << ", ";
    //     }
    //     cout << endl;
    // }
    logr("Success");
  }
  
  // get the number of entries in the TChain
  Int_t GetEntries() {
    return nEvents;
  }
  
  /// CUTS
  ///
  
  bool Cut(bool expression, Cuts::CutType cutName) {
    cutValues[cutName] = expression ? 1 : 0;
    return expression;
  }
  
  bool Cut(Cuts::CutType cutName) {
    return cutValues[cutName];
  }
  
  bool CutsRange(int start, int end) {
    return std::all_of(cutValues.begin() + start, cutValues.begin() + end, [](int i){return i > 0;});
  }
  
  void InitCuts() {
    std::fill(cutValues.begin(), cutValues.end(), -1);
  }
  
  void PrintCuts() {
    print(&cutValues);
  }
  
  void UpdateCutFlow() {
    size_t i = 0;
    CutFlow[0]++;
    while (i < cutValues.size() && cutValues[i] > 0)
      CutFlow[++i]++;
  }
  
  void PrintCutFlow() {
    int fn = 20;
    int ns = 6 + int(log10(CutFlow[0]));
    int n = 10;
    
    log();
    cout << std::setprecision(2) << std::fixed;
    cout << LOG_PREFIX << setw(fn) << "CutFlow" << setw(ns) << "N" << setw(n) << "Abs Eff" << setw(n) << "Rel Eff" << endl;
    cout << LOG_PREFIX << string(fn + ns + n*2, '=') << endl;
    cout << LOG_PREFIX << setw(fn) << "None" << setw(ns) << CutFlow[0] << setw(n) << 100.0 << setw(n) << 100.0 << endl;
    
    int i = 1;
    for (auto elt : Cuts::CutName) {
      cout << LOG_PREFIX << std::setw(fn) << elt.second << std::setw(ns) << CutFlow[i] << std::setw(n) << 100.*float(CutFlow[i])/float(CutFlow[0]) << std::setw(n) << 100.*float(CutFlow[i])/float(CutFlow[i - 1]) << endl;
      i++;
    }
  }
  
  void SaveCutFlow() {
    TH1F *CutFlowHist = new TH1F("h_CutFlow","CutFlow", Cuts::CutName.size(), -0.5, Cuts::CutName.size() - 0.5);
    CutFlowHist->SetBinContent(1, CutFlow[0]);
    CutFlowHist->GetXaxis()->SetBinLabel(1, "no selection");
    int i = 1;
    for (auto elt : Cuts::CutName) {
      CutFlowHist->SetBinContent(i + 1, CutFlow[elt.first]);
      CutFlowHist->GetXaxis()->SetBinLabel(i + 1, elt.second.c_str());
      i++;
    }
    CutFlowHist->Write();
    
    std::ofstream f(outputdir + "/" + sample + "_cutflow.txt");
    if (f.is_open()) {
      vector<string> cutNames;
      for (auto elt : Cuts::CutName) {
        cutNames.push_back(elt.second);
      }
      WriteVector(f, CutFlow);
      WriteVector(f, cutNames);
      f.close();
    }
  }
  
  template<typename t>
  void WriteVector(std::ostream & out, vector<t> & vec, string delimiter=", ") {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
      out << vec[i] << delimiter;
    }
    out << vec.back() << endl;
  }
  
  // void PrintAllCuts() {
  //     log("CUTS:");
  //     for (size_t i = 0; i < savedCuts.size(); ++i)
  //         print(&savedCuts[i]);
  // }
  
  /// HISTOGRAMS
  ///
  
  size_t AddHist(Hists::HistType ht, string name="", string title="", int bins=10, double min=0., double max=1.) {
    size_t i = hists.size();
    TH1F* newHist = new TH1F(name.c_str(), title.c_str(), bins, min, max);
    hists.push_back(newHist);
    histIndex[ht] = i;
    return i;
  }
  
  void Fill(Hists::HistType ht, double value) {
    hists[histIndex[ht]]->Fill(value);
  }
  
  void WriteHists() {
    for (size_t i = 0; i < hists.size(); ++i)
      hists[i]->Write();
  }
  
  void UpdateSelectionIndex(size_t entry) {
    chain->GetN(entry);
    selectionIndex[chain->currentTree].push_back(chain->currentEntry);
  }
  
  void WriteSelectionIndex() {
    std::ofstream f(outputdir + "/" + sample + "_selection.txt");
    log(selectionIndex.size());
    for (auto elt : selectionIndex) {
      log(elt.size());
    }
    if (f.is_open()) {
      for (size_t i = 0; i < selectionIndex.size(); i++){
        f << outputTrees[i] << ": ";
        for (size_t j = 0; j < selectionIndex[i].size(); j++) {
          f << selectionIndex[i][j] << " ";
        }
        f << endl;
      }
      f.close();
    }
  }
  
  /// SWITCHES, TIMING, AND LOGGING
  ///
  
  // Turn on or off debug logging with this switch
  void Debug(bool debugSwitch) {
    debug = debugSwitch;
  }
  
  // turn on or off timing logs with this switch (dependent of debug=true)
  void Timing(bool timingSwitch) {
    timing=timingSwitch;
  }
  
  // prints a summary of the current entry
  void Current() {
    log();
    if (varIndex.size() > 0) {
      log();
      print("SINGLE VARIABLES:");
    }
    for (auto it = varIndex.begin(); it != varIndex.end(); it++) {
      print(it->first, 1);
      print(varValues[it->second], 2);
    }
    if (vectorVarIndex.size() > 0) {
      log();
      print("VECTOR VARIABLES:");
    }
    for (auto it = vectorVarIndex.begin(); it != vectorVarIndex.end(); it++) {
      print(it->first, 1);
      print(varValues[it->second], 2);
    }
    if (MapVectors.size() > 0) {
      log();
      print("MAP VECTORS:");
    }
    for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
      if (subIndex[it->second].second == vectorType::Map) {
        print(it->first, 1);
        print(MapVectors[subIndex[it->second].first], 2);
      }
    }
    if (MockVectors.size() > 0) {
      log();
      print("MOCK VECTORS:");
    }
    for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
      if (subIndex[it->second].second == vectorType::Mock) {
        print(it->first, 1);
        print(MockVectors[subIndex[it->second].first], 2);
      }
    }
    if (LorentzVectors.size() > 0) {
      log();
      print("TLORENTZ VECTORS:");
    }
    for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
      if (subIndex[it->second].second == vectorType::Lorentz) {
        print(it->first, 1);
        print(LorentzVectors[subIndex[it->second].first], 2);
      }
    }
    log();
    log();
  }
  
  // time of last call, in seconds
  double ts() {
    return duration/1000000.;
  }
  
  // '', in milliseconds
  double tms() {
    return duration/1000.;
  }
  
  // '', in microseconds
  double tus() {
    return duration;
  }
  
  // log the time! of the last call
  void logt() {
    if (timing)
      log("(execution time: " + to_string(ts()) + "s)");
  }
  
  // internal timer start
  void start() {
    tStart(timestart);
  }
  
  // internal timer end
  void end() {
    duration = tEnd(timestart);
  }
  
  /// PUBLIC DATA
  ///
  // general init vars, parsed from argv
  string sample, inputspec, outputdir;
  
  // number of events
  Int_t nEvents, nMin, nMax;
  // internal debug switch
  bool debug=true, timing=true, saveCuts=true;
  
  vector<int> CutFlow = vector<int>(Cuts::COUNT + 1, 0);
  int last = 1;
  
private:
  /// CON/DESTRUCTOR HELPERS
  ///
  template<typename t>
  void DelVector(vector<vector<t*>> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
      DelVector(v[i]);
    }
  }
  
  template<typename t>
  void DelVector(vector<t*> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
      delete v[i];
      v[i] = nullptr;
    }
  }
  
  /// VARIABLE TRACKER HELPERS
  ///
  
  void AddCompsBase(string& vectorName, vector<string>& components) {
    if(compIndex.find(vectorName) != compIndex.end())
      throw "Vector variable '" + vectorName + "' already exists!";
    size_t index = compIndex.size();
    logp("Adding " + to_string(components.size()) + " components to vector " + vectorName + "...  ");
    compVectors.push_back(vector<vector<TLeaf*>>());
    compNames.push_back(vector<string>());
    // cout << endl;
    for (size_t i = 0; i < components.size(); ++i) {
      auto inp = chain->FindLeaf(components[i].c_str());
      // cout << i << " " << inp.size() << endl;
      compVectors[index].push_back(inp);
      compNames[index].push_back(lastWord(components[i]));
    }
    // cout << endl;
    // cout << compVectors[index][0].size()  << endl;
    // cout << compVectors[index].size() << endl;
    // cout << compVectors.size() << endl;
    compIndex[vectorName] = index;
  }
  
  /// ENTRY LOADER HELPERS
  ///
  
  void SetLorentz(size_t leafIndex, size_t lvIndex, size_t treeIndex) {
    vector<vector<TLeaf*>> & v = compVectors[leafIndex];
    vector<TLorentzVector> * ret = LorentzVectors[lvIndex];
    
    ret->clear();
    size_t n = v[0][treeIndex]->GetLen();
    // cout << endl << n << v[1]->GetLen() << v[2]->GetLen() << v[3]->GetLen() << endl;
    for (size_t i = 0; i < n; ++i) {
      ret->push_back(TLorentzVector());
      ret->at(i).SetPtEtaPhiM(
                              v[0][treeIndex]->GetValue(i),
                              v[1][treeIndex]->GetValue(i),
                              v[2][treeIndex]->GetValue(i),
                              v[3][treeIndex]->GetValue(i)
                              );
    }
  }
  
  void SetMock(size_t leafIndex, size_t mvIndex, size_t treeIndex) {
    vector<vector<TLeaf*>> & v = compVectors[leafIndex];
    vector<LorentzMock>* ret = MockVectors[mvIndex];
    ret->clear();
    
    size_t n = v[0][treeIndex]->GetLen(), size = v.size();
    
    for(size_t i = 0; i < n; ++i) {
      switch(size) {
        case 2: {
          ret->push_back(LorentzMock(v[0][treeIndex]->GetValue(i), v[1][treeIndex]->GetValue(i)));
          break;
        }
        case 3: {
          ret->push_back(LorentzMock(v[0][treeIndex]->GetValue(i), v[1][treeIndex]->GetValue(i), v[2][treeIndex]->GetValue(i)));
          break;
        }
        case 4: {
          ret->push_back(LorentzMock(v[0][treeIndex]->GetValue(i), v[1][treeIndex]->GetValue(i), v[2][treeIndex]->GetValue(i), v[3][treeIndex]->GetValue(i)));
          break;
        }
        default: {
          throw "Invalid number arguments for MockTLorentz vector (" + to_string(size) + ")";
        }
      }
    }
  }
  
  void SetMap(size_t leafIndex, size_t mIndex, size_t treeIndex) {
    vector<vector<TLeaf*>> & v = compVectors[leafIndex];
    
    // cout << mIndex << endl;
    // cout << leafIndex << endl;
    // cout << MapVectors.size() << endl;
    vector<vector<double>>* ret = MapVectors[mIndex];
    size_t n = v[0][treeIndex]->GetLen();
    
    ret->clear();
    ret->resize(n);
    
    for (size_t i = 0; i < n; ++i) {
      ret->at(i).clear();
      ret->at(i).reserve(v.size());
      for (size_t j = 0; j < v.size(); ++j) {
        ret->at(i).push_back(v[j][treeIndex]->GetValue(i));
      }
    }
  }
  
  void SetVar(size_t leafIndex, size_t treeIndex) {
    *varValues[leafIndex] = varLeaves[leafIndex][treeIndex]->GetValue(0);
  }
  
  void SetVectorVar(size_t leafIndex, size_t treeIndex) {
    vectorVarValues[leafIndex]->clear();
    for (int i = 0; i < vectorVarLeaves[leafIndex][treeIndex]->GetLen(); ++i) {
      vectorVarValues[leafIndex]->push_back(vectorVarLeaves[leafIndex][treeIndex]->GetValue(i));
    }
    // log(leafIndex);
    // log(vectorVarLeaves[leafIndex]->GetLen());
    // log(vectorVarValues[leafIndex].size());
    // log();
  }
  
  /// SWITCH, TIMING, AND LOGGING HELPERS
  ///
  
  double tsRaw(double d) {
    return d/1000000.;
  }
  
  void tStart(std::chrono::high_resolution_clock::time_point & t) {
    t = std::chrono::high_resolution_clock::now();
  }
  
  double tEnd(std::chrono::high_resolution_clock::time_point & t) {
    return duration_cast<microseconds>(std::chrono::high_resolution_clock::now() - t).count();
  }
  
  void log() {
    if (debug)
      cout << LOG_PREFIX << endl;
  }
  
  template<typename t>
  void log(t s) {
    if (debug) {
      cout << LOG_PREFIX;
      lograw(s);
      cout << endl;
    }
  }
  
  template<typename t>
  void logp(t s) {
    if (debug) {
      cout << LOG_PREFIX;
      lograw(s);
    }
  }
  
  template<typename t>
  void logr(t s) {
    if (debug) {
      lograw(s);
      cout << endl;
    }
  }
  
  template<typename t>
  void warning(t s) {
    debug = true;
    log("WARNING :: " + to_string(s));
    debug = false;
  }
  
  template<typename t>
  void lograw(t s) {
    cout << s;
  }
  
  void indent(int level){
    cout << LOG_PREFIX << string(level*3, ' ');
  }
  
  void print(string s, int level=0) {
    indent(level);
    cout << s << endl;
  }
  
  template<typename t>
  void print(t* var, int level=0) {
    indent(level); cout << *var << endl;
  }
  
  template<typename t>
  void print(vector<t>* var, int level=0) {
    indent(level);
    cout << "{ ";
    for (size_t i = 0; i < var->size() - 1; ++i) {
      cout << var->at(i) << ", ";
    }
    cout << var->back() << " }";
    cout << endl;
  }
  
  template<typename t>
  void print(vector<vector<t>>* var, int level=0) {
    for (size_t i = 0; i < var->size(); ++i) {
      print(&var[i], level);
    }
  }
  
  void print(vector<LorentzMock>* var, int level=0) {
    for (size_t i = 0; i < var->size(); ++i) {
      auto elt = var->at(i);
      indent(level); cout << "(Pt,Eta)=(" << elt.Pt() << "," << elt.Eta() << "}" << endl;
    }
  }
  
  void print(vector<TLorentzVector>* var, int level=0) {
    for (size_t i = 0; i < var->size(); ++i) {
      auto elt = var->at(i);
      indent(level);
      elt.Print();
    }
  }
  
  void print() {
    indent(0);
    cout << endl;
  }
  
  vector<string> split(string s, char delimiter = '.') {
    std::replace(s.begin(), s.end(), delimiter, ' ');
    vector<string> ret;
    stringstream ss(s);
    string temp;
    while(ss >> temp)
      ret.push_back(temp);
    return ret;
  }
  
  string lastWord(string s, char delimiter = '.') {
    return split(s, delimiter).back();
  }
  
  /// PRIVATE DATA
  ///
  // general entry
  int currentEntry;
  
  // histogram data
  vector<TH1F*> hists;
  vector<size_t> histIndex = vector<size_t>(Hists::COUNT);
  
  // timing data
  double duration = 0;
  std::chrono::high_resolution_clock::time_point timestart, programstart;
  
  // file data
  ParallelTreeChain *chain=nullptr;
  TFile *file=nullptr;
  vector<string> outputTrees;
  
  // logging data
  const string LOG_PREFIX = "SVJselection :: ";
  std::map<vectorType, std::string> componentTypeStrings = {
    {vectorType::Lorentz, "TLorentzVector"},
    {vectorType::Mock, "MockTLorentzVector"},
    {vectorType::Map, "Map"}
  };
  
  // single variable data
  std::map<string, size_t> varIndex;
  vector<vector<TLeaf *>> varLeaves; // CHANGE
  vector<double*> varValues;
  
  // vector variable data
  std::map<string, size_t> vectorVarIndex;
  vector<vector<TLeaf *>> vectorVarLeaves; // CHANGE
  vector<vector<double>*> vectorVarValues;
  
  // vector component data
  //   indicies
  std::map<string, size_t> compIndex;
  vector<pair<size_t, vectorType>> subIndex;
  //   names
  vector<vector<vector<TLeaf*>>> compVectors; // CHANGE
  vector<vector<string>> compNames;
  //   values
  vector< vector< TLorentzVector >*> LorentzVectors;
  vector< vector< LorentzMock >*> MockVectors;
  vector<vector<vector<double>>*> MapVectors;
  
  // cut variables
  vector<int> cutValues = vector<int>(Cuts::COUNT, -1);
  vector<vector<size_t>> selectionIndex;
};
