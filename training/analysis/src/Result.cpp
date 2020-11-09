//
//  Result.cpp
//  drawAucDistributions
//
//  Created by Jeremi Niedziela on 02/11/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "Result.hpp"

void Result::print()
{
  cout<<name<<"\t"<<mass<<"\t"<<r_inv<<"\t"<<AUC<<endl;
}

void Epoch::print()
{
  cout<<"Learning rate: "<<learningRate<<"\ttraining loss: "<<trainingLoss<<"\tvalidation loss: "<<validationLoss<<endl;
}
