//
//  ModelStats.hpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef ModelStats_hpp
#define ModelStats_hpp

#include "Helpers.hpp"
#include "Result.hpp"

class ModelStats {
public:
  ModelStats(){}
  
  double getAucAverageOverAllSignals() const;
  
  struct sortByAucAverageOverAllSignals{
      inline bool operator() (const ModelStats& m1, const ModelStats& m2){
          return (m1.getAucAverageOverAllSignals() > m2.getAucAverageOverAllSignals());
      }
  };
  
  string aucsFileName;
  string lossesFileName;
  
  vector<Result> results; /// AUCs for all mass -- r_inv combinations
  vector<Epoch> epochs; /// Losses and learning rates per epoch
};

#endif /* ModelStats_hpp */
