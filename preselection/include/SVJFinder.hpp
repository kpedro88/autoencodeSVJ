#include "Helpers.hpp"
#include "ParallelTreeChain.h"
#include "LorentzMock.h"

//using std::fabs;
//using std::chrono::microseconds;
//using std::chrono::duration_cast;
//using std::string;
//using std::endl;
//using std::cout;
//using std::vector;
//using std::pair;
//using std::to_string;
//using std::stringstream;
//using std::setw;


class SVJFinder {
public:
  /// CON/DESTRUCTORS
  ///
  
  // constructor, requires argv as input
  SVJFinder(char **argv);
  
  // destructor for dynamically allocated data
  ~SVJFinder();
  
  /// FILE HANDLERS
  ///
  
  // sets up paralleltreechain and returns a pointer to it
  ParallelTreeChain* MakeChain();
  
  /// VARIABLE TRACKER FUNCTIONS
  ///
  
  // creates, assigns, and returns tlorentz vector pointer to be updated on GetEntry
  vector<TLorentzVector>* AddLorentz(string vectorName, vector<string> components);
  
  // creates, assigns, and returns mock tlorentz vector pointer to be updated on GetEntry
  vector<LorentzMock>* AddLorentzMock(string vectorName, vector<string> components);
  
  // creates, assigns, and returns general double vector pointer to be updated on GetEntry
  vector<vector<double>>* AddComps(string vectorName, vector<string> components);
  
  // creates, assigns, and returns a vectorized single variable pointer to be updates on GetEntry
  vector<double>* AddVectorVar(string vectorVarName, string component);
  
  // creates, assigns, and returns a singular double variable pointer to update on GetEntry
  double* AddVar(string varName, string component);
  
  /// ENTRY LOADING
  ///
  
  void reloadLeaves() {}
  
  // get the ith entry of the TChain
  void GetEntry(int entry = 0);
  
  // get the number of entries in the TChain
  inline Int_t GetEntries() { return nEvents; }
  
  /// CUTS
  ///
  
  bool Cut(bool expression, CutType cutName);
  
  inline bool Cut(CutType cutName) { return cutValues[int(cutName)]; }
  
  bool CutsRange(int start, int end);
  
  inline void InitCuts() { std::fill(cutValues.begin(), cutValues.end(), -1); }
  inline void PrintCuts() { print(&cutValues); }
  
  void UpdateCutFlow();
  
  void PrintCutFlow();
  
  void SaveCutFlow();
  
  template<typename t>
  void WriteVector(std::ostream & out, vector<t> & vec, string delimiter=", ");
  
  /// HISTOGRAMS
  ///
  
  size_t AddHist(HistType ht, string name="", string title="", int bins=10, double min=0., double max=1.);
  
  inline void Fill(HistType ht, double value) { hists[histIndex[size_t(ht)]]->Fill(value); }
  
  void WriteHists();
  
  void UpdateSelectionIndex(size_t entry);
  
  void WriteSelectionIndex();
  
  /// SWITCHES, TIMING, AND LOGGING
  ///
  
  // Turn on or off debug logging with this switch
  inline void Debug(bool debugSwitch) { debug = debugSwitch; }
  
  // turn on or off timing logs with this switch (dependent of debug=true)
  inline void Timing(bool timingSwitch) { timing=timingSwitch; }
  
  // prints a summary of the current entry
  void Current();
  
  // time of last call, in seconds
  double ts() { return duration/1000000.; }
  
  // '', in milliseconds
  double tms() { return duration/1000.; }
  
  // '', in microseconds
  double tus() { return duration; }
  
  // log the time! of the last call
  void logt() { if (timing) log("(execution time: " + to_string(ts()) + "s)"); }
  
  
  /// PUBLIC DATA
  ///
  // general init vars, parsed from argv
  string sample, inputspec, outputdir;
  
  // number of events
  Int_t nEvents, nMin, nMax;
  // internal debug switch
  bool debug=true, timing=true, saveCuts=true;
  
  vector<int> CutFlow = vector<int>(int(CutType::COUNT) + 1, 0);
  int last = 1;
  
private:
  /// CON/DESTRUCTOR HELPERS
  ///
  template<typename t>
  void DelVector(vector<vector<t*>> &v);
  
  template<typename t>
  void DelVector(vector<t*> &v);
  
  /// VARIABLE TRACKER HELPERS
  ///
  
  void AddCompsBase(string& vectorName, vector<string>& components);
  
  /// ENTRY LOADER HELPERS
  ///
  
  void SetLorentz(size_t leafIndex, size_t lvIndex, size_t treeIndex);
  
  void SetMock(size_t leafIndex, size_t mvIndex, size_t treeIndex);
  
  void SetMap(size_t leafIndex, size_t mIndex, size_t treeIndex);
  
  void SetVar(size_t leafIndex, size_t treeIndex);
  
  void SetVectorVar(size_t leafIndex, size_t treeIndex);
  
  /// SWITCH, TIMING, AND LOGGING HELPERS
  ///
  
  double tsRaw(double d) { return d/1000000.; }
    
  template<typename t>
  void log(t s);
  
  template<typename t>
  void logp(t s) { if (debug) { cout<<s; } }
  
  template<typename t>
  void logr(t s) {
    if (debug) {
      cout << s;
      cout << endl;
    }
  }
  
  template<typename t>
  void warning(t s) {
    debug = true;
    log("WARNING :: " + to_string(s));
    debug = false;
  }
  
  void indent(int level){
    cout << string(level*3, ' ');
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
  vector<size_t> histIndex = vector<size_t>(size_t(HistType::COUNT));
  
  // timing data
  double duration = 0;
  std::chrono::high_resolution_clock::time_point timestart, programstart;
  
  // file data
  ParallelTreeChain *chain=nullptr;
  TFile *file=nullptr;
  vector<string> outputTrees;
  
  // logging data
  
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
  vector<int> cutValues = vector<int>(int(CutType::COUNT), -1);
  vector<vector<size_t>> selectionIndex;
};
