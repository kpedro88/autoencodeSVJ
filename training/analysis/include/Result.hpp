//
//  Result.hpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Result_hpp
#define Result_hpp

#include "Helpers.hpp"

class Result {
public:
  Result(){}
 
  string name;
  double mass, r_inv, AUC;
  
  void print();
};

class Epoch {
public:
  Epoch(){}
  
  void print();
  
  double trainingLoss, learningRate, validationLoss;
};

#endif /* Result_hpp */
