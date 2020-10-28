//
//  Jet.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Jet_hpp
#define Jet_hpp

#include "Helpers.hpp"

class Jet{
public:
  Jet(){}
  
  vector<double> EFPs;
  double eta, phi, pt, mass, chargedFraction, PTD, axis2, flavor, energy;
  
  void print(){
    cout<<"Jet:"<<endl;
    cout<<"\teta: "<<eta<<"\tphi: "<<phi<<"\tpt: "<<pt<<"\tmass: "<<mass<<"\tcharged fraction: "<<chargedFraction;
    cout<<"\tPTD: "<<PTD<<"\taxis2: "<<axis2<<"\tflavor: "<<flavor<<"\tenergy: "<<energy<<endl;
    cout<<"\tEFPs:";
    for(double EFP : EFPs) cout<<EFP<<"\t";
    cout<<endl;
  }
};

#endif /* Jet_hpp */
