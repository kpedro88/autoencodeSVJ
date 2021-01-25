#include "SVJFinder.hpp"
#include "LorentzMock.h"

namespace Vetos {
bool LeptonVeto(LorentzMock& lepton) {
  return fabs(lepton.Pt()) > 10 && fabs(lepton.Eta()) < 2.4;
}

bool IsolationVeto(double &iso) {
  return iso >= 0.4;
}

bool JetEtaVeto(TLorentzVector& jet) {
  return abs(jet.Eta()) < 2.4;
}

bool JetDeltaEtaVeto(TLorentzVector& jet1, TLorentzVector& jet2) {
  return abs(jet1.Eta() - jet2.Eta()) < 1.5;
}

bool JetPtVeto(TLorentzVector& jet) {
  return jet.Pt() > 200.;
}
}

size_t leptonCount(vector<LorentzMock>* leptons, vector<double>* isos) {
  size_t n = 0;
  // size_t lepton_size = std::min(leptons->size(), isos->size());
  for (size_t i = 0; i < leptons->size(); ++i)
  if (Vetos::LeptonVeto(leptons->at(i)) && Vetos::IsolationVeto(isos->at(i)))
    n++;
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
  // core.MakeFileCollection();
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
  vector<LorentzMock>* Electrons = core.AddLorentzMock("Electron", {"Electron.PT","Electron.Eta"});
  vector<LorentzMock>* Muons = core.AddLorentzMock("Muon", {"MuonLoose.PT", "MuonLoose.Eta"});
  vector<double>* MuonIsolation = core.AddVectorVar("MuonIsolation", "MuonLoose.IsolationVarRhoCorr");
  vector<double>* ElectronIsolation = core.AddVectorVar("ElectronIsolation", "Electron.IsolationVarRhoCorr");
  double* metFull_Pt = core.AddVar("metMET", "MissingET.MET");
  double* metFull_Phi = core.AddVar("metPhi", "MissingET.Phi");
  
  // loop over the first nEntries (debug)
  // start loop timer
  
  auto start = now();
  
  
  for (Int_t entry = core.nMin; entry < core.nMax; ++entry) {
    
    // init
    core.InitCuts();
    
    core.GetEntry(entry);
    
    // require zero leptons which pass cuts
    // pre lepton cut
    core.Fill(HistType::pre_lep, Muons->size() + Electrons->size());
    
    // made it here
    
    core.Cut(
             (leptonCount(Muons, MuonIsolation) + leptonCount(Electrons, ElectronIsolation)) < 1,
             CutType::leptonCounts
             );
    
    // didn't make it here
    
    
    
    if (!core.Cut(CutType::leptonCounts)) {
      core.UpdateCutFlow();
      continue;
    }
    
    
    core.Fill(HistType::post_lep, Muons->size() + Electrons->size());
    
    
    // require more than 1 jet
    core.Cut(
             Jets->size() > 1,
             CutType::jetCounts
             );
    
    
    
    // rest of cuts, dependent on jetcount
    if (core.Cut(CutType::jetCounts)) {
      
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
      core.Cut(
               Vetos::JetEtaVeto(Jets->at(0)) && Vetos::JetEtaVeto(Jets->at(1)),
               CutType::jetEtas
               );
      
      // leading jets meet delta eta veto
      core.Cut(
               Vetos::JetDeltaEtaVeto(Jets->at(0), Jets->at(1)),
               CutType::jetDeltaEtas
               );
      
      // ratio between calculated mt2 of dijet system and missing momentum is not negligible
      core.Cut(
               ((*metFull_Pt) / MT2) > 0.15,
               CutType::metRatio
               );
      
      // require both leading jets to have transverse momentum greater than 200
      core.Fill(HistType::pre_1pt, Jets->at(0).Pt());
      core.Fill(HistType::pre_2pt, Jets->at(1).Pt());
      
      core.Cut(
               Vetos::JetPtVeto(Jets->at(0)) && Vetos::JetPtVeto(Jets->at(1)),
               CutType::jetPt
               );
      if (!core.Cut(CutType::jetPt)) {
        core.UpdateCutFlow();
        continue;
      }
      
      core.Fill(HistType::post_1pt, Jets->at(0).Pt());
      core.Fill(HistType::post_2pt, Jets->at(1).Pt());
      
      // conglomerate cut, whether jet is a dijet
      core.Cut(
               core.Cut(CutType::jetEtas) && core.Cut(CutType::jetPt),
               CutType::jetDiJet
               );
      
      // magnitude of MT > 1500
      core.Cut(
               MT2 > 1500,
               CutType::metValue
               );
      
      // tighter MET/MT ratio
      core.Cut(
               ((*metFull_Pt) / MT2) > 0.25,
               CutType::metRatioTight
               );
      
      // final selection cut
      core.Cut(
               core.CutsRange(0, int(CutType::selection)) && core.Cut(CutType::metRatioTight),
               CutType::selection
               );
      
      // save histograms, if passing
      if (core.Cut(CutType::selection)) {
        core.UpdateSelectionIndex(entry);
        core.Fill(HistType::dEta, fabs(Jets->at(0).Eta() - Jets->at(1).Eta()));
        core.Fill(HistType::dPhi, fabs(deltaPhi(Jets->at(0).Phi(), Jets->at(1).Phi())));
        core.Fill(HistType::tRatio, (*metFull_Pt) / MT2);
        core.Fill(HistType::mjj, Vjj.M());
        core.Fill(HistType::met2, MT2);
        core.Fill(HistType::metPt, *metFull_Pt);
        
      }
      
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
