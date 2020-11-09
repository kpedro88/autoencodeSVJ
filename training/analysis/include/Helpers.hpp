//
//  Helpers.hpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Helpers_h
#define Helpers_h

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <locale>

#pragma clang diagnostic push // save the current state
#pragma clang diagnostic ignored "-Wdocumentation" // turn off ROOT's warnings
#pragma clang diagnostic ignored "-Wconversion"

#include "TH1D.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TApplication.h"
#include "TGraph.h"

#pragma clang diagnostic pop // restores the saved state for diagnostics

using namespace std;

/**
 Forces printing numbers with commas as decimal separators instead of dots
 */
inline void useCommaAsDecimalSeparator()
{
  cout.imbue(locale("de_DE"));
}

/**
 Returns vector of file names in provided path matching given pattern
 */
inline vector<string> getFileInPathMatchingPattern(string path, string pattern)
{
  DIR* dirp = opendir(path.c_str());
  struct dirent * dp;

  vector<string> filePaths;
  
  while((dp = readdir(dirp))){
    string fileName = dp->d_name;
    
    if(fileName.find(pattern) != string::npos){
      filePaths.push_back(fileName);
    }
  }
  closedir(dirp);
  
  return filePaths;
}

/**
 Converts provided string to vector of strings, splitting input by found commas
 */
inline vector<string> splitByComma(string input)
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


/**
 Converts any numerical value to string with given precision
 */
template <typename T>
string to_string_with_precision(const T a_value, const int n = 6)
{
  ostringstream out;
  out.precision(n);
  out << fixed << a_value;
  return out.str();
}

#endif /* Helpers_h */
