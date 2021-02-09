#include "SVJFinder.hpp"
#include "LorentzMock.h"

const int maxNleptons = 0;
const double minLeptonPt = 10; // GeV
const double maxLeptonEta = 2.4;
const double minLeptonIsolation = 0.4;

const int minNjets = 2;
const double maxJetEta = 2.4;
const double minJetPt = 200; //GeV
const double maxJetDeltaEta = 1.5;
const double minJetMt = 0; //GeV
const double minMetRatio = 0;


int leptonCount(vector<LorentzMock>* leptons)
{
  int n = 0;
  for(int i=0; i<leptons->size(); i++) {
    if(fabs(leptons->at(i).Pt()) >= minLeptonPt &&
       fabs(leptons->at(i).Eta()) <= maxLeptonEta &&
       leptons->at(i).Isolation() >= minLeptonIsolation) n++;
  }
  return n;
}

int main(int argc, char **argv)
{
  if(argc != 6){
    cout<<"Usage:"<<endl;
    cout<<"./SVJselection input_file_list sample_name output_dir first_event last_event"<<endl;
    exit(0);
  }
  
  
  // declare core object and enable debug
  SVJFinder core(argv);
  
  // make file collection and chain
  core.MakeChain();
  
  // add histogram tracking
  core.AddHist(HistType::dEta, "h_dEta", "#Delta#eta(j0,j1)", 100, 0, 10);
  core.AddHist(HistType::dPhi, "h_dPhi", "#Delta#Phi(j0,j1)", 100, 0, 5);
  core.AddHist(HistType::tRatio,  "h_transverseratio", "MET/M_{T}", 100, 0, 1);
  core.AddHist(HistType::met2, "h_Mt", "m_{T}", 750, 0, 7500);
  core.AddHist(HistType::mjj, "h_Mjj", "m_{JJ}", 750, 0, 7500);
  core.AddHist(HistType::metPt, "h_METPt", "MET_{p_{T}}", 100, 0, 2000);
  
  // histograms for pre/post PT wrt PT cut (i.e. after MET, before PT && afer PT)
  core.AddHist(HistType::pre_1pt, "h_pre_1pt", "pre PT cut leading jet pt", 100, 0, 2500);
  core.AddHist(HistType::pre_2pt, "h_pre_2pt", "pre PT cut subleading jet pt", 100, 0, 2500);
  core.AddHist(HistType::post_1pt, "h_post_1pt", "post PT cut leading jet pt", 100, 0, 2500);
  core.AddHist(HistType::post_2pt, "h_post_2pt", "post PT cut subleading jet pt", 100, 0, 2500);
  
  // histograms for pre/post lepton count wrt lepton cut
  core.AddHist(HistType::pre_lep, "h_pre_lep", "lepton count pre-cut", 10, 0, 10);
  core.AddHist(HistType::post_lep, "h_post_lep", "lepton count post-cut", 10, 0, 10);
  
  // mt2 pre cut
  core.AddHist(HistType::pre_MT, "h_pre_MT", "pre-cut m_{T}", 750, 0, 7500);
  core.AddHist(HistType::pre_mjj, "h_pre_Mjj", "pre-cut m_{JJ}", 750, 0, 7500);
  
  // add componenets for jets (tlorentz)
  
  vector<TLorentzVector>* Jets = core.AddLorentz("Jet", {"Jet.PT","Jet.Eta","Jet.Phi","Jet.Mass"});
  vector<LorentzMock>* Electrons = core.AddLorentzMock("Electron", {"Electron.PT","Electron.Eta", "Electron.IsolationVarRhoCorr"});
  vector<LorentzMock>* Muons = core.AddLorentzMock("Muon", {"MuonLoose.PT", "MuonLoose.Eta", "MuonLoose.IsolationVarRhoCorr"});
  double* metFull_Pt = core.AddVar("metMET", "MissingET.MET");
  double* metFull_Phi = core.AddVar("metPhi", "MissingET.Phi");
  
  // loop over the first nEntries (debug)
  // start loop timer
  
  auto start = now();
  
  
  for (Int_t entry = core.nMin; entry < core.nMax; ++entry) {
    
    core.GetEntry(entry);
    
    // require zero leptons which pass cuts
    // pre lepton cut
    core.Fill(HistType::pre_lep, Muons->size() + Electrons->size());
    
    // made it here
    bool passesNleptons = (leptonCount(Muons) + leptonCount(Electrons)) <= maxNleptons;
    core.SetCutValue(passesNleptons, CutType::leptonCounts);
    
    if (!passesNleptons){
      core.UpdateCutFlow();
      continue;
    }
    
    core.Fill(HistType::post_lep, Muons->size() + Electrons->size());
    
    bool passesNjets = Jets->size() >= minNjets;
    
    for(int iJet=1; iJet<Jets->size(); iJet++){
      if(Jets->at(iJet).Pt() > Jets->at(iJet-1).Pt()){
        cout<<"ERROR -- jets don't seem to be ordered by pt!!"<<endl;
        exit(1);
      }
    }
    
    core.SetCutValue(passesNjets, CutType::jetCounts);
    
    // rest of cuts, dependent on jetcount
    if(!passesNjets) {
      core.UpdateCutFlow();
      continue;;
    }
    
    if(Jets->size() < 2){
      cout<<"WARNING -- less than 2 jets in the event -> this wasn't implemented yet... skipping."<<endl;
      continue;
    }
    
    TLorentzVector Vjj = Jets->at(0) + Jets->at(1);
    double metFull_Py = (*metFull_Pt)*sin(*metFull_Phi);
    double metFull_Px = (*metFull_Pt)*cos(*metFull_Phi);
    double Mjj = Vjj.M(); // SAVE
    double Mjj2 = Mjj*Mjj;
    double ptjj = Vjj.Pt();
    double ptjj2 = ptjj*ptjj;
    double ptMet = Vjj.Px()*metFull_Px + Vjj.Py()*metFull_Py;
    double MT2 = sqrt(Mjj2 + 2*(sqrt(Mjj2 + ptjj2)*(*metFull_Pt) - ptMet)); // SAVE
    
    // fill pre-cut MT2 histogram
    core.Fill(HistType::pre_MT, MT2);
    core.Fill(HistType::pre_mjj, Mjj);
    
    // leading jet etas both meet eta veto
    bool passesJetEta = fabs(Jets->at(0).Eta()) <= maxJetEta && fabs(Jets->at(0).Eta()) <= maxJetEta;
    core.SetCutValue(passesJetEta, CutType::jetEtas);
    
    // leading jets meet delta eta veto
    bool passesJetDeltaEta = fabs(Jets->at(0).Eta() - Jets->at(1).Eta()) <= maxJetDeltaEta;
    core.SetCutValue(passesJetDeltaEta, CutType::jetDeltaEtas);
    
    // ratio between calculated mt2 of dijet system and missing momentum is not negligible
    bool passesMetRatio = ((*metFull_Pt) / MT2) >= minMetRatio;
    core.SetCutValue(passesMetRatio, CutType::metRatio);
    
    // require both leading jets to have transverse momentum greater than 200
    core.Fill(HistType::pre_1pt, Jets->at(0).Pt());
    core.Fill(HistType::pre_2pt, Jets->at(1).Pt());
    
    
    bool passesJetPt = Jets->at(0).Pt() >= minJetPt && Jets->at(1).Pt() >= minJetPt;
    core.SetCutValue(passesJetPt, CutType::jetPt);
    
    if(!passesJetPt || !passesJetEta){
      core.UpdateCutFlow();
      continue;
    }
    
    core.Fill(HistType::post_1pt, Jets->at(0).Pt());
    core.Fill(HistType::post_2pt, Jets->at(1).Pt());
    
    bool passesMt = MT2 >= minJetMt;
    core.SetCutValue(passesMt, CutType::metValue);
    
    // final selection cut
    bool passesAllSelections = core.PassesAllSelections();
    core.SetCutValue(passesAllSelections, CutType::selection);
    
    // save histograms, if passing
    if(passesAllSelections) {
      core.UpdateSelectionIndex(entry);
      core.Fill(HistType::dEta, fabs(Jets->at(0).Eta() - Jets->at(1).Eta()));
      core.Fill(HistType::dPhi, fabs(deltaPhi(Jets->at(0).Phi(), Jets->at(1).Phi())));
      core.Fill(HistType::tRatio, (*metFull_Pt) / MT2);
      core.Fill(HistType::mjj, Vjj.M());
      core.Fill(HistType::met2, MT2);
      core.Fill(HistType::metPt, *metFull_Pt);
      
    }
    
    core.UpdateCutFlow();
  }
  
  cout<<"Time elapsed: "<<duration(start, now())<<endl;
  
  core.WriteHists();
  core.WriteSelectionIndex();
  core.SaveCutFlow();
  core.PrintCutFlow();
  
  return 0;
}
