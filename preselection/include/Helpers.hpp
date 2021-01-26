//
//  Helpers.hpp
//  xSVJselection
//
//  Created by Jeremi Niedziela on 25/01/2021.
//

#ifndef Helpers_h
#define Helpers_h



#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <string>
#include "THashList.h"
#include "TBenchmark.h"
#include <sstream>
#include <fstream>
#include <utility>
#include <map>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <math.h>

#pragma clang diagnostic push // save the current state
#pragma clang diagnostic ignored "-Wdocumentation" // turn off ROOT's warnings
#pragma clang diagnostic ignored "-Wconversion"

#include "TTree.h"
#include "TLeaf.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TLorentzVector.h"
#include "TH1F.h"
#include "TMath.h"
#include "Rtypes.h"


#pragma clang diagnostic pop // restores the saved state for diagnostics

using namespace std;

enum vectorType {
  Lorentz,
  Mock,
  Map
};

enum class CutType {
  leptonCounts = 0,
  jetCounts = 1,
  jetEtas,
  jetDeltaEtas,
  jetPt,
  metValue,
  metRatio,
  selection,
};

const vector<CutType> cutTypes = {
  CutType::leptonCounts,
  CutType::jetCounts,
  CutType::jetEtas,
  CutType::jetDeltaEtas,
  CutType::jetPt,
  CutType::metValue,
  CutType::metRatio,
  CutType::selection,
};

const map<CutType, string> CutName {
  {CutType::leptonCounts, "0 Passing Leptons"},
  {CutType::jetCounts, "n Jets > 1"},
  {CutType::jetEtas, "abs jet Etas < 2.4"},
  {CutType::jetDeltaEtas, "abs DeltaEta < 1.5"},
  {CutType::jetPt, "Jet PT > 200"},
  {CutType::metValue, "M_T > 1500"},
  {CutType::metRatio, "MET/M_T > 0.25"},
  {CutType::selection, "final selection"}
};

enum class HistType {
  dEta = 0,
  dPhi = 1,
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

/// Returns delta phi reduced to [-pi, pi]
inline double deltaPhi(double phi1, double phi2)
{
  double x = phi1 - phi2;
  constexpr double o2pi = 1. / (2. * M_PI);
  if (std::abs(x) <= double(M_PI)) return x;
  double n = std::round(x * o2pi);
  return x - n * double(2. * M_PI);
}

/// Calculate duration between two events
/// \param t0 Start time
/// \param t1 End time
/// \return Difference between events t0 and t1 in seconds
template<class T>
double duration(T t0,T t1)
{
  auto elapsed_secs = t1-t0;
  typedef std::chrono::duration<float> float_seconds;
  auto secs = std::chrono::duration_cast<float_seconds>(elapsed_secs);
  return secs.count();
}

/// Returns current time
inline std::chrono::time_point<std::chrono::steady_clock> now()
{
  return std::chrono::steady_clock::now();
}


template<typename t>
void WriteVector(ostream & out, vector<t> & vec, string delimiter=", ")
{
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    out << vec[i] << delimiter;
  }
  out << vec.back() << endl;
}

inline string lastWord(string s)
{
  std::replace(s.begin(), s.end(), '.', ' ');
  vector<string> ret;
  stringstream ss(s);
  string temp;
  while(ss >> temp) ret.push_back(temp);
  return ret.back();
}

#endif /* Helpers_h */
