
void readDelphes()
{
  TFile *inFile = TFile::Open("/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/qcd/qcd_sqrtshatTeV_13TeV_PU20_9.root");
  
  auto tree = (TTree*)inFile->Get("Delphes");
  
  
  int n_jets;
  
  tree->SetBranchAddress("Jet_size", &n_jets);
  
  
  
  float jet_eta[9999], jet_phi[9999];
  
  tree->SetBranchAddress("Jet.Eta", &jet_eta);
  tree->SetBranchAddress("Jet.Phi", &jet_phi);
  
  
  for(int iEvent=0; iEvent<10; iEvent++){
    tree->GetEntry(iEvent);
    cout<<"\n\nEvent: "<<iEvent<<endl;
    
    for(int iJet=0; iJet<n_jets; iJet++){
      cout<<"eta: "<<jet_eta[iJet]<<endl;
    }
  }
  
  
  return;
}

HDF5 "test_nanoAOD.h5" {
GROUP "/" {
   GROUP "event_features" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 4, 5 ) / ( 4, 5 ) }
         DATA {
         (0,0): 12.9878, 0, 1.63086, 281.936, 268.014,
         (1,0): 86.572, 0, -0.624023, 200.96, 33.2164,
         (2,0): 19.9272, 0, -0.74231, 223.405, 207.714,
         (3,0): 44.1704, 0, -0.956909, 109.791, 64.9488
         }
      }
      DATASET "labels" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 5 ) / ( 5 ) }
         DATA {
         (0): "MET", "METEta", "METPhi", "MT", "Mjj"
         }
      }
   }
}
}
