//
//  EventProcessor.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef EventProcessor_hpp
#define EventProcessor_hpp

#include "Helpers.hpp"
#include "Event.hpp"

class EventProcessor{
public:
  EventProcessor(){}
  
  
  vector<string> getLabels(Group group)
  {
    DataSet dataset = group.openDataSet("labels");
    
    vector<string> labels;
    
    if(dataset.getTypeClass() != H5T_STRING){
      cout<<"ERROR -- labels should be of type string, but they are not"<<endl;
      return labels;
    }
    
    DataSpace dataspace = dataset.getSpace();
    StrType strType = dataset.getStrType();

    hssize_t nPoints = dataspace.getSimpleExtentNpoints();
    size_t size = strType.getSize();
    
    
    char values[nPoints][size];
    dataset.read(values, strType, dataspace);
    
    for(int i=0; i<nPoints; i++){
      string label = "";
      for(int j=0; j<size; j++){
        label += values[i][j];
      }
      
      label.erase(remove(label.begin(), label.end(), '\0'), label.end());
      labels.push_back(label);
    }
    return labels;
  }

  vector<shared_ptr<Event>> getValues(Group groupEvent, Group groupEFPs, Group groupJet)
  {
    vector<shared_ptr<Event>> events;

    // Initialize events with global features
    DataSet datasetEvent = groupEvent.openDataSet("data");

    if(datasetEvent.getTypeClass() != H5T_FLOAT){
      cout<<"ERROR -- data should be of type float, but they are not"<<endl;
      return events;
    }
    
    DataSpace dataspaceEvent = datasetEvent.getSpace();
    FloatType floatTypeEvent = datasetEvent.getFloatType();
    
    hsize_t dimEvent[2];
    dataspaceEvent.getSimpleExtentDims(dimEvent, NULL);
    
    double values[dimEvent[0]][dimEvent[1]];
    datasetEvent.read(values, floatTypeEvent, dataspaceEvent);
    
    for(int i=0; i<dimEvent[0]; i++){
      auto event = make_shared<Event>();
      
      event->MET    = values[i][0];
      event->METeta = values[i][1];
      event->METphi = values[i][2];
      event->MT     = values[i][3];
      event->Mjj    = values[i][4];
      
      events.push_back(event);
    }
    
    // Add jets with their EFPs to events
    DataSet datasetEFPs = groupEFPs.openDataSet("data");

    if(datasetEFPs.getTypeClass() != H5T_FLOAT){
      cout<<"ERROR -- data should be of type float, but they are not"<<endl;
      return events;
    }
    
    DataSpace dataspaceEFPs = datasetEFPs.getSpace();

    hsize_t dim[3];
    dataspaceEFPs.getSimpleExtentDims(dim, NULL);
    hsize_t memdim = dim[0] * dim[1] * dim[2];

    vector<float> valuesEFP(memdim);

    datasetEFPs.read(valuesEFP.data(), PredType::NATIVE_FLOAT, dataspaceEFPs, dataspaceEFPs);
    
    for(int iEvent=0; iEvent<dim[0]; iEvent++){
      for(int iJet=0; iJet<dim[1]; iJet++){
        auto jet = make_shared<Jet>();
        
        vector<double> variablesForJet;
        
        for(int iVar=0; iVar<dim[2]; iVar++){
          variablesForJet.push_back(valuesEFP[iEvent*dim[1]*dim[2] + iJet*dim[2] + iVar]);
        }
        jet->EFPs = variablesForJet;
        events.at(iEvent)->jets.push_back(jet);
      }
    }
    
    // Add basic information to jets
     DataSet datasetJets = groupJet.openDataSet("data");

     if(datasetJets.getTypeClass() != H5T_FLOAT){
       cout<<"ERROR -- data should be of type float, but they are not"<<endl;
       return events;
     }
     
     DataSpace dataspaceJets = datasetJets.getSpace();

     hsize_t dimJets[3];
     dataspaceJets.getSimpleExtentDims(dimJets, NULL);
     hsize_t memdimJets = dimJets[0] * dimJets[1] * dimJets[2];

     vector<float> valuesJets(memdimJets);
     datasetJets.read(valuesJets.data(), PredType::NATIVE_FLOAT, dataspaceJets, dataspaceJets);
     
     for(int iEvent=0; iEvent<dimJets[0]; iEvent++){
       for(int iJet=0; iJet<dimJets[1]; iJet++){
         auto jet = events[iEvent]->jets[iJet];
         
         jet->eta             = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 0];
         jet->phi             = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 1];
         jet->pt              = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 2];
         jet->mass            = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 3];
         jet->chargedFraction = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 4];
         jet->PTD             = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 5];
         jet->axis2           = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 6];
         jet->flavor          = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 7];
         jet->energy          = valuesJets[iEvent*dimJets[1]*dimJets[2] + iJet*dimJets[2] + 8];
       }
     }
    
    return events;
  }

  
  
};




#endif /* EventProcessor_hpp */
