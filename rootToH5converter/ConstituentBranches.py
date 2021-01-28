from InputTypes import *

class ConstituentBranches:
    
    def __init__(self, tree, input_type):
        self.track_eta = []
        self.track_phi = []
        self.track_pt = []
        self.track_mass = []
        self.track_jet_index = []
    
        self.neutral_hadron_eta = []
        self.neutral_hadron_phi = []
        self.neutral_hadron_pt = []
        self.neutral_hadron_mass = []
    
        self.photon_eta = []
        self.photon_phi = []
        self.photon_pt = []
        self.photon_mass = []
    
        if input_type == InputTypes.Delphes:
            self.track_eta = tree["EFlowTrack/EFlowTrack.Eta"].array()
            self.track_phi = tree["EFlowTrack/EFlowTrack.Phi"].array()
            self.track_pt = tree["EFlowTrack/EFlowTrack.PT"].array()
            self.track_mass = 0
        
            self.neutral_hadron_eta = tree["EFlowNeutralHadron/EFlowNeutralHadron.Eta"].array()
            self.neutral_hadron_phi = tree["EFlowNeutralHadron/EFlowNeutralHadron.Phi"].array()
            self.neutral_hadron_pt = tree["EFlowNeutralHadron/EFlowNeutralHadron.ET"].array()
            self.neutral_hadron_mass = 0
        
            self.photon_eta = tree["Photon/Photon.Eta"].array()
            self.photon_phi = tree["Photon/Photon.Phi"].array()
            self.photon_pt = tree["Photon/Photon.PT"].array()
            self.photon_mass = 0
    
        elif input_type == InputTypes.nanoAOD:
            print("WARNING -- handling of tracks for nanoAOD not implemented!!!")
            print("WARNING -- handling of neutral hadrons for nanoAOD not implemented!!!")
        
            self.photon_eta = tree["Photon_eta"].array()
            self.photon_phi = tree["Photon_phi"].array()
            self.photon_pt = tree["Photon_pt"].array()
            self.photon_mass = tree["Photon_mass"].array()
    
        elif input_type == InputTypes.PFnanoAOD:
        
            print("WARNING -- handling of neutral hadrons for PFnanoAOD not implemented!!!")
        
            self.track_eta = tree["JetPFCands_eta"].array()
            self.track_phi = tree["JetPFCands_phi"].array()
            self.track_pt = tree["JetPFCands_pt"].array()
            self.track_mass = tree["JetPFCands_mass"].array()
            self.track_jet_index = tree["JetPFCands_jetIdx"].array()
        
            self.photon_eta = tree["Photon_eta"].array()
            self.photon_phi = tree["Photon_phi"].array()
            self.photon_pt = tree["Photon_pt"].array()
            self.photon_mass = tree["Photon_mass"].array()