
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
