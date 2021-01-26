#include "Helpers.hpp"
#include "ParallelTreeChain.h"
#include "LorentzMock.h"

class SVJFinder {
public:
  // constructor, requires argv as input
  SVJFinder(char **argv);
  
  // destructor for dynamically allocated data
  ~SVJFinder();
  
  // sets up paralleltreechain and returns a pointer to it
  ParallelTreeChain* MakeChain();
  
  // creates, assigns, and returns tlorentz vector pointer to be updated on GetEntry
  vector<TLorentzVector>* AddLorentz(string vectorName, vector<string> components);
  
  // creates, assigns, and returns mock tlorentz vector pointer to be updated on GetEntry
  vector<LorentzMock>* AddLorentzMock(string vectorName, vector<string> components);
  
  // creates, assigns, and returns a singular double variable pointer to update on GetEntry
  double* AddVar(string varName, string component);
  
  // get the ith entry of the TChain
  void GetEntry(int entry = 0);
  

  void SetCutValue(bool expression, CutType cutName);
  
  bool PassesAllSelections();
  
  void UpdateCutFlow();
  
  void PrintCutFlow();
  
  void SaveCutFlow();
  
  /// HISTOGRAMS
  ///
  
  size_t AddHist(HistType ht, string name="", string title="", int bins=10, double min=0., double max=1.);
  
  inline void Fill(HistType ht, double value) { hists[histIndex[size_t(ht)]]->Fill(value); }
  
  void WriteHists();
  
  void UpdateSelectionIndex(size_t entry);
  
  void WriteSelectionIndex();
  
  
  /// PUBLIC DATA
  ///
  // general init vars, parsed from argv
  string sample, inputspec, outputdir;
  
  // number of events
  Int_t nEvents, nMin, nMax;
  
  vector<int> CutFlow = vector<int>(cutTypes.size() + 1, 0);
  int last = 1;
  
private:
  void AddCompsBase(string& vectorName, vector<string>& components);
  void SetLorentz(size_t leafIndex, size_t lvIndex, size_t treeIndex);
  void SetMock(size_t leafIndex, size_t mvIndex, size_t treeIndex);
  void SetMap(size_t leafIndex, size_t mIndex, size_t treeIndex);
  void SetVar(size_t leafIndex, size_t treeIndex);
  
  void SetVectorVar(size_t leafIndex, size_t treeIndex);

  
  int currentEntry; // general entry
  
  // histogram data
  vector<TH1F*> hists;
  vector<size_t> histIndex = vector<size_t>(size_t(HistType::COUNT));
  
  // file data
  ParallelTreeChain *chain=nullptr;
  TFile *file=nullptr;
  vector<string> outputTrees;
  
  // single variable data
  map<string, size_t> varIndex;
  vector<vector<TLeaf *>> varLeaves; // CHANGE
  vector<double*> varValues;
  
  // vector variable data
  map<string, size_t> vectorVarIndex;
  vector<vector<TLeaf *>> vectorVarLeaves; // CHANGE
  vector<vector<double>*> vectorVarValues;
  
  // vector component data
  //   indicies
  map<string, size_t> compIndex;
  vector<pair<size_t, vectorType>> subIndex;
  
  //   names
  vector<vector<vector<TLeaf*>>> compVectors; // CHANGE
  vector<vector<string>> compNames;
  
  //   values
  vector< vector< TLorentzVector >*> LorentzVectors;
  vector< vector< LorentzMock >*> MockVectors;
  vector<vector<vector<double>>*> MapVectors;
  
  // cut variables
  map<CutType, bool> cutValues;
  vector<vector<size_t>> selectionIndex;
};
