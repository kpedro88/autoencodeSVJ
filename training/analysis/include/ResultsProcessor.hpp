//
//  ResultsProcessor.hpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef ResultsProcessor_hpp
#define ResultsProcessor_hpp

#include "Helpers.hpp"
#include "ModelStats.hpp"

class ResultsProcessor {
public:
  ResultsProcessor(){}
  
  /// Loads training results from all files in given path matching given pattern
  vector<ModelStats> getModelStatsFromPathMarchingPatter(string aucsPath, string resultsPath, string filePattern);
  
  
  void sortModelsByAucAverageOverAllSignals(vector<ModelStats> &models);
  
  
private:
  /// Loads training results from given path
  vector<Result> getResultsFromFile(string inFilePath);
  
  /// Loads training evolution (losses vs. epoch) from given path
  vector<Epoch> getEpochsFromFile(string inFilePath);
  
  
  
  
};

#endif /* ResultsProcessor_hpp */
