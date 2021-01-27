import numpy as np
from Jet import Jet
from PhysObject import PhysObject

class Event:
    def __init__(self, tree, input_type, iEvent, constituentBranches, delta_r):
    
        if input_type != "Delphes" and input_type != "nanoAOD" and input_type != "PFnanoAOD":
            print("\n\nERROR -- Event: unknown input type!\n\n")
            exit(0)
    
        if input_type == "Delphes":
            self.met = tree["MissingET.MET"].array()[iEvent][0]
            self.metPhi = tree["MissingET.Phi"].array()[iEvent][0]
            self.metEta = tree["MissingET.Eta"].array()[iEvent][0]

            self.nJets = tree["Jet_size"].array()[iEvent]
            self.nTracks = tree["EFlowTrack_size"].array()[iEvent]
            self.nNeutralHadrons = tree["EFlowNeutralHadron_size"].array()[iEvent]
            self.nPhotons = tree["Photon_size"].array()[iEvent]
            
        elif input_type == "nanoAOD":
            self.met = tree["MET_pt"].array()[iEvent]
            self.metPhi = tree["MET_phi"].array()[iEvent]
            self.metEta = 0

            self.nJets = tree["nJet"].array()[iEvent]
            self.nTracks = 0
            self.nNeutralHadrons = 0
            self.nPhotons = tree["nPhoton"].array()[iEvent]
            
        elif input_type == "PFnanoAOD":
            self.met = tree["MET_pt"].array()[iEvent]
            self.metPhi = tree["MET_phi"].array()[iEvent]
            self.metEta = 0
    
            self.nJets = tree["nJet"].array()[iEvent]
            self.nTracks = tree["nJetPFCands"].array()[iEvent]
            self.nNeutralHadrons = 0
            self.nPhotons = tree["nPhoton"].array()[iEvent]
            
        self.Mjj = 99999
        self.MT = 99999
    
        
        
        self.tracks = []
        for iTrack in range(0, self.nTracks):
            mass = constituentBranches.track_mass[iEvent][iTrack] if hasattr(constituentBranches.track_mass, "__getitem__") else 0
            self.tracks.append(PhysObject(eta = constituentBranches.track_eta[iEvent][iTrack],
                                          phi = constituentBranches.track_phi[iEvent][iTrack],
                                          pt = constituentBranches.track_pt[iEvent][iTrack],
                                          mass = mass))

        
        self.neutral_hadrons = []
        for iNeutralHadron in range(0, self.nNeutralHadrons):
            mass = constituentBranches.neutral_hadron_mass[iEvent][iNeutralHadron] if hasattr(constituentBranches.neutral_hadron_mass, "__getitem__") else 0
            self.neutral_hadrons.append(PhysObject(eta = constituentBranches.neutral_hadron_eta[iEvent][iNeutralHadron],
                                          phi = constituentBranches.neutral_hadron_phi[iEvent][iNeutralHadron],
                                          pt = constituentBranches.neutral_hadron_pt[iEvent][iNeutralHadron],
                                          mass = mass))

        
        self.photons = []
        for iPhoton in range(0, self.nPhotons):
            mass = constituentBranches.photon_mass[iEvent][iPhoton] if hasattr(constituentBranches.photon_mass, "__getitem__") else 0
            self.photons.append(PhysObject(eta = constituentBranches.photon_eta[iEvent][iPhoton],
                                          phi = constituentBranches.photon_phi[iEvent][iPhoton],
                                          pt = constituentBranches.photon_pt[iEvent][iPhoton],
                                          mass = mass))

        
        self.jets = []
        for iJet in range(0, self.nJets):
            self.jets.append(Jet(tree, input_type, iEvent, iJet))
    
        for iJet, jet in enumerate(self.jets):
            jet.fill_constituents(self.tracks, self.neutral_hadrons, self.photons, delta_r, iJet, constituentBranches.track_jet_index[iEvent])
    
        if self.nJets >= 2:
            self.calculate_internals()
    
    def print(self):
        print("\nEvent features: ")
        print(Event.get_features_names())
        print(self.get_features())
        
        print("\nnJets:", self.nJets)
        for i, jet in enumerate(self.jets):
            print("\tjet ", i, " n constituents: ", len(jet.constituents))
        
        print("nTracks:", self.nTracks)
        print("nPhotons:", self.nPhotons)
        print("nNeutral hadrons:", self.nNeutralHadrons)
    
    def has_jets_with_no_constituents(self, max_n_jets):
        has_jets_with_no_constituents = False
    
        for i in range(0, max_n_jets):
            if len(self.jets[i].constituents) == 0:
                has_jets_with_no_constituents = True
                break
                
        return has_jets_with_no_constituents
    
    def calculate_internals(self):
        
        if len(self.jets) <= 1:
            print("ERROR -- events has less than 2 jets, which should never happen!")
        
        if len(self.jets) != 2:
            print("ERROR -- expected two jets in the event, but there are ", len(self.jets))
        
        Vjj = self.jets[0].get_four_vector() + self.jets[1].get_four_vector()
        met_py = self.met * np.sin(self.metPhi)
        met_px = self.met * np.cos(self.metPhi)
    
        Mjj = Vjj.M()
        Mjj2 = Mjj * Mjj
        ptjj = Vjj.Pt()
        ptjj2 = ptjj * ptjj
        ptMet = Vjj.Px() * met_px + Vjj.Py() * met_py
    
        MT = np.sqrt(Mjj2 + 2. * (np.sqrt(Mjj2 + ptjj2) * self.met - ptMet))
        
        self.Mjj = Mjj
        self.MT = MT

    def get_features(self):
        return [
            self.met,
            self.metEta,
            self.metPhi,
            self.MT,
            self.Mjj,
        ]
    
    @staticmethod
    def get_features_names():
        return [
            'MET',
            'METEta',
            'METPhi',
            'MT',
            'Mjj',
        ]