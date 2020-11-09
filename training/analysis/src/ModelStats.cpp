//
//  ModelStats.cpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "ModelStats.hpp"

double ModelStats::getAucAverageOverAllSignals() const
{
  double averageAUC = 0;
  
  for(Result result : results){
    averageAUC += result.AUC;
  }
  
  averageAUC /= results.size();
  
  return averageAUC;
}
