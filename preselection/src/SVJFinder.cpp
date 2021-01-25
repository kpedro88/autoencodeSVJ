//
//  SVJFinder.cpp
//  xSVJselection
//
//  Created by Jeremi Niedziela on 25/01/2021.
//

#include "SVJFinder.hpp"

SVJFinder::SVJFinder(char **argv)
{
  inputspec = argv[1];
  cout<< "File list to open: " << inputspec << endl;
  
  sample = argv[2];
  cout << "Sample name: " << sample << endl;
  
  outputdir = argv[3];
  cout << "Output directory: " << outputdir << endl;
  
  nMin = std::stoi(argv[4]);
  nMax = std::stoi(argv[5]);
  
  if (nMin < 0) nMin = 0;
}

SVJFinder::~SVJFinder()
{
  cout << "Quitting; cleaning up class variables...  ";
  
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
  cout << "Success" << endl;
}

ParallelTreeChain* SVJFinder::MakeChain() {
  
  cout << "Creating file chain with tree type 'Delphes'...";
  chain = new ParallelTreeChain();
  outputTrees = chain->GetTrees(inputspec, "Delphes");
  
  selectionIndex.resize(outputTrees.size());
  
  file = new TFile((outputdir + "/" + sample + "_output.root").c_str(), "RECREATE");
  nEvents = (Int_t)chain->GetEntries();
  
  if (nMax < 0 || nMax > nEvents) nMax = nEvents;
  
  cout << "Success" << endl;
  return chain;
}

vector<TLorentzVector>* SVJFinder::AddLorentz(string vectorName, vector<string> components)
{
  assert(components.size() == 4);
  AddCompsBase(vectorName, components);
  size_t i = LorentzVectors.size();
  subIndex.push_back(std::make_pair(i, vectorType::Lorentz));
  vector<TLorentzVector>* ret = new vector<TLorentzVector>;
  LorentzVectors.push_back(ret);
  return ret;
}

vector<LorentzMock>* SVJFinder::AddLorentzMock(string vectorName, vector<string> components)
{
  
  assert(components.size() > 1 && components.size() < 5);
  AddCompsBase(vectorName, components);
  size_t i = MockVectors.size();
  subIndex.push_back(std::make_pair(i, vectorType::Mock));
  vector<LorentzMock>* ret = new vector<LorentzMock>;
  MockVectors.push_back(ret);
  return ret;
}

vector<vector<double>>* SVJFinder::AddComps(string vectorName, vector<string> components)
{
  
  AddCompsBase(vectorName, components);
  size_t i = MapVectors.size();
  subIndex.push_back(std::make_pair(i, vectorType::Map));
  vector<vector<double>>* ret = new vector<vector<double>>;
  MapVectors.push_back(ret);
  return ret;
}

vector<double>* SVJFinder::AddVectorVar(string vectorVarName, string component)
{
  cout << "Adding 1 component to vector var " << vectorVarName << "...  ";
  int i = int(vectorVarValues.size());
  vectorVarIndex[vectorVarName] = i;
  vectorVarLeaves.push_back(chain->FindLeaf(component));
  vector<double>* ret = new vector<double>;
  vectorVarValues.push_back(ret);
  cout << "Success" << endl;
  return ret;
}

double* SVJFinder::AddVar(string varName, string component)
{
  cout << "Adding 1 component to var " << varName << "...  ";
  size_t i = varLeaves.size();
  varIndex[varName] = i;
  double* ret = new double;
  varLeaves.push_back(chain->FindLeaf(component));
  varValues.push_back(ret);
  cout << "Success" << endl;
  return ret;
}

void SVJFinder::GetEntry(int entry)
{
  assert(entry < chain->GetEntries());
  cout << "Getting entry " << to_string(entry) << "...  ";
  int treeId = chain->GetEntry(entry);
  currentEntry = entry;
  if (chain->currentEntry == 0) {
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

  cout << "Success" << endl;
}

bool SVJFinder::Cut(bool expression, CutType cutName)
{
  cutValues[int(cutName)] = expression ? 1 : 0;
  return expression;
}

bool SVJFinder::CutsRange(int start, int end)
{
  return std::all_of(cutValues.begin() + start, cutValues.begin() + end, [](int i){return i > 0;});
}

void SVJFinder::UpdateCutFlow()
{
  size_t i = 0;
  CutFlow[0]++;
  while (i < cutValues.size() && cutValues[i] > 0)
    CutFlow[++i]++;
}

void SVJFinder::PrintCutFlow()
{
  int fn = 20;
  int ns = 6 + int(log10(CutFlow[0]));
  int n = 10;
  
  
  cout << std::setprecision(2) << std::fixed;
  cout << setw(fn) << "CutFlow" << setw(ns) << "N" << setw(n) << "Abs Eff" << setw(n) << "Rel Eff" << endl;
  cout << string(fn + ns + n*2, '=') << endl;
  cout << setw(fn) << "None" << setw(ns) << CutFlow[0] << setw(n) << 100.0 << setw(n) << 100.0 << endl;
  
  int i = 1;
  for (auto elt : CutName) {
    cout << std::setw(fn) << elt.second << std::setw(ns) << CutFlow[i] << std::setw(n) << 100.*float(CutFlow[i])/float(CutFlow[0]) << std::setw(n) << 100.*float(CutFlow[i])/float(CutFlow[i - 1]) << endl;
    i++;
  }
}

void SVJFinder::SaveCutFlow()
{
  TH1F *CutFlowHist = new TH1F("h_CutFlow","CutFlow", int(CutName.size()), -0.5, CutName.size() - 0.5);
  CutFlowHist->SetBinContent(1, CutFlow[0]);
  CutFlowHist->GetXaxis()->SetBinLabel(1, "no selection");
  int i = 1;
  for (auto elt : CutName) {
    CutFlowHist->SetBinContent(i + 1, CutFlow[int(elt.first)]);
    CutFlowHist->GetXaxis()->SetBinLabel(i + 1, elt.second.c_str());
    i++;
  }
  CutFlowHist->Write();
  
  std::ofstream f(outputdir + "/" + sample + "_cutflow.txt");
  if (f.is_open()) {
    vector<string> cutNames;
    for (auto elt : CutName) {
      cutNames.push_back(elt.second);
    }
    WriteVector(f, CutFlow);
    WriteVector(f, cutNames);
    f.close();
  }
}



size_t SVJFinder::AddHist(HistType ht, string name, string title, int bins, double min, double max)
{
  size_t i = hists.size();
  TH1F* newHist = new TH1F(name.c_str(), title.c_str(), bins, min, max);
  hists.push_back(newHist);
  histIndex[size_t(ht)] = i;
  return i;
}

void SVJFinder::WriteHists()
{
  for (size_t i = 0; i < hists.size(); ++i)
  hists[i]->Write();
}

void SVJFinder::UpdateSelectionIndex(size_t entry)
{
  chain->GetN(entry);
  selectionIndex[chain->currentTree].push_back(chain->currentEntry);
}

void SVJFinder::WriteSelectionIndex()
{
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

//void SVJFinder::Current()
//{
//
//  if (varIndex.size() > 0) {
//
//    print("SINGLE VARIABLES:");
//  }
//  for (auto it = varIndex.begin(); it != varIndex.end(); it++) {
//    print(it->first, 1);
//    print(varValues[it->second], 2);
//  }
//  if (vectorVarIndex.size() > 0) {
//
//    print("VECTOR VARIABLES:");
//  }
//  for (auto it = vectorVarIndex.begin(); it != vectorVarIndex.end(); it++) {
//    print(it->first, 1);
//    print(varValues[it->second], 2);
//  }
//  if (MapVectors.size() > 0) {
//
//    print("MAP VECTORS:");
//  }
//  for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
//    if (subIndex[it->second].second == vectorType::Map) {
//      print(it->first, 1);
//      print(MapVectors[subIndex[it->second].first], 2);
//    }
//  }
//  if (MockVectors.size() > 0) {
//
//    print("MOCK VECTORS:");
//  }
//  for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
//    if (subIndex[it->second].second == vectorType::Mock) {
//      print(it->first, 1);
//      print(MockVectors[subIndex[it->second].first], 2);
//    }
//  }
//  if (LorentzVectors.size() > 0) {
//
//    print("TLORENTZ VECTORS:");
//  }
//  for (auto it = compIndex.begin(); it != compIndex.end(); it++) {
//    if (subIndex[it->second].second == vectorType::Lorentz) {
//      print(it->first, 1);
//      print(LorentzVectors[subIndex[it->second].first], 2);
//    }
//  }
//
//
//}


template<typename t>
void SVJFinder::DelVector(vector<vector<t*>> &v)
{
  for (size_t i = 0; i < v.size(); ++i) {
    DelVector(v[i]);
  }
}

template<typename t>
void SVJFinder::DelVector(vector<t*> &v)
{
  for (size_t i = 0; i < v.size(); ++i) {
    delete v[i];
    v[i] = nullptr;
  }
}

void SVJFinder::AddCompsBase(string& vectorName, vector<string>& components)
{
  if(compIndex.find(vectorName) != compIndex.end()){
    throw "Vector variable '" + vectorName + "' already exists!";
  }
  size_t index = compIndex.size();
  cout << "Adding " << to_string(components.size()) << " components to vector " << vectorName << "...  " << endl;
  compVectors.push_back(vector<vector<TLeaf*>>());
  compNames.push_back(vector<string>());
  
  for (size_t i = 0; i < components.size(); ++i) {
    auto inp = chain->FindLeaf(components[i].c_str());
    compVectors[index].push_back(inp);
    compNames[index].push_back(lastWord(components[i]));
  }
  
  compIndex[vectorName] = index;
}

void SVJFinder::SetLorentz(size_t leafIndex, size_t lvIndex, size_t treeIndex)
{
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

void SVJFinder::SetMock(size_t leafIndex, size_t mvIndex, size_t treeIndex)
{
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

void SVJFinder::SetMap(size_t leafIndex, size_t mIndex, size_t treeIndex)
{
  vector<vector<TLeaf*>> & v = compVectors[leafIndex];
  
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

void SVJFinder::SetVar(size_t leafIndex, size_t treeIndex)
{
  *varValues[leafIndex] = varLeaves[leafIndex][treeIndex]->GetValue(0);
}

void SVJFinder::SetVectorVar(size_t leafIndex, size_t treeIndex)
{
  vectorVarValues[leafIndex]->clear();
  for (int i = 0; i < vectorVarLeaves[leafIndex][treeIndex]->GetLen(); ++i) {
    vectorVarValues[leafIndex]->push_back(vectorVarLeaves[leafIndex][treeIndex]->GetValue(i));
  }
}
