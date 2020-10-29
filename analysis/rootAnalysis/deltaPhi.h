#ifndef DataFormats_Math_deltaPhi_h
#define DataFormats_Math_deltaPhi_h

/* function to compute deltaPhi
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 * stabilize range reduction
 */

#include <cmath>

// reduce to [-pi,pi]
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

template <typename T>
constexpr T deltaPhi(T phi1, T phi2)
{
  return reduceRange(phi1 - phi2);
}

#endif
