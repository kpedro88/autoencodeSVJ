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

/// reduce to [-pi,pi]
template <typename T>
T reduceRange(T x)
{
  constexpr T o2pi = 1. / (2. * M_PI);
  if (std::abs(x) <= T(M_PI)) return x;
  T n = std::round(x * o2pi);
  return x - n * T(2. * M_PI);
}

double  deltaPhi(double phi1, double phi2)  { return reduceRange(phi1 - phi2);                  }
double  deltaPhi(float phi1, double phi2)   { return deltaPhi(static_cast<double>(phi1), phi2); }
double  deltaPhi(double phi1, float phi2)   { return deltaPhi(phi1, static_cast<double>(phi2)); }
float   deltaPhi(float phi1, float phi2)    { return reduceRange(phi1 - phi2);                  }

template <typename T1, typename T2>
constexpr auto deltaPhi(T1 const& t1, T2 const& t2) -> decltype(deltaPhi(t1.phi(), t2.phi()))
{
  return deltaPhi(t1.phi(), t2.phi());
}

/**
 Function to compute deltaPhi.
 Ported from original code in RecoJets by Fedor Ratnikov, FNAL stabilize range reduction.
 */
template <typename T>
constexpr T deltaPhi(T phi1, T phi2)
{
  return reduceRange(phi1 - phi2);
}

#endif /* Helpers_h */
