//
//  Event.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Event_hpp
#define Event_hpp

#include "Helpers.hpp"
#include "Jet.hpp"

class Event{
public:
  Event(){}
  
  double MET, METeta, METphi, MT, Mjj;
  
  vector<shared_ptr<Jet>> jets;
  
  void print(){
    cout<<"Event:"<<endl;
    cout<<"\tMET: "<<MET<<"\tMETeta: "<<METeta<<"\tMETphi: "<<METphi;
    cout<<"\tMT: "<<MT<<"\tMjj: "<<Mjj<<endl;
    cout<<"\tJets:"<<endl;
    for(auto jet : jets) jet->print();
  }
};

#endif /* Event_hpp */
