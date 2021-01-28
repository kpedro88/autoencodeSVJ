from enum import Enum

class InputTypes(Enum):
    Delphes = 0,
    nanoAOD = 1,
    PFnanoAOD = 2

class DataProcessor:
    def __init__(self, tree, input_type):
        
        if not input_type in InputTypes:
            print("\n\nERROR -- DataProcessor: unknown input type: ", input_type,"\n\n")
            exit(0)
        
        self.input_type = input_type
        self.tree = tree
        
        self.variables = {
            InputTypes.Delphes: {
                # number of objects
                "N_jets": "Jet_size",
                "N_tracks": "EFlowTrack_size",
                "N_neutral_hadrons": "EFlowNeutralHadron_size",
                "N_photons": "Photon_size",
                # event features
                "MET_pt": "MissingET/MissingET.MET",
                "MET_eta": "MissingET/MissingET.Eta",
                "MET_phi": "MissingET/MissingET.Phi",
                # jet features
                "Jet_eta": "Jet/Jet.Eta",
                "Jet_phi": "Jet/Jet.Phi",
                "Jet_pt": "Jet/Jet.PT",
                "Jet_mass": "Jet/Jet.Mass",
                "Jet_nCharged": "Jet/Jet.NCharged",
                "Jet_nNeutral": "Jet/Jet.NNeutrals",
                "Jet_flavor": "Jet/Jet.Flavor",
                # tracks
                "Track_eta": "EFlowTrack/EFlowTrack.Eta",
                "Track_phi": "EFlowTrack/EFlowTrack.Phi",
                "Track_pt": "EFlowTrack/EFlowTrack.PT",
                # neutral hadrons
                "Neutral_eta": "EFlowNeutralHadron/EFlowNeutralHadron.Eta",
                "Neutral_phi": "EFlowNeutralHadron/EFlowNeutralHadron.Phi",
                "Neutral_pt": "EFlowNeutralHadron/EFlowNeutralHadron.ET",
                # photons
                "Photon_eta": "Photon/Photon.Eta",
                "Photon_phi": "Photon/Photon.Phi",
                "Photon_pt": "Photon/Photon.PT",
                
            },
            InputTypes.nanoAOD: {
                # number of objects
                "N_jets": "nJet",
                "N_photons": "nPhoton",
                # event features
                "MET_pt": "MET_pt",
                "MET_phi": "MET_phi",
                # jet features
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
                # photons
                "Photon_eta": "Photon_eta",
                "Photon_phi": "Photon_phi",
                "Photon_pt": "Photon_pt",
                "Photon_mass": "Photon_mass",
                
            },
            InputTypes.PFnanoAOD: {
                # number of objects
                "N_jets": "nJet",
                "N_tracks": "nJetPFCands",
                "N_photons": "nPhoton",
                # event features
                "MET_pt": "MET_pt",
                "MET_phi": "MET_phi",
                # jet features
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
                # tracks
                "Track_eta": "JetPFCands_eta",
                "Track_phi": "JetPFCands_phi",
                "Track_pt": "JetPFCands_pt",
                "Track_mass": "JetPFCands_mass",
                "Track_jet_index": "JetPFCands_jetIdx",
                # photons
                "Photon_eta": "Photon_eta",
                "Photon_phi": "Photon_phi",
                "Photon_pt": "Photon_pt",
                "Photon_mass": "Photon_mass",
            }
        }

    def get_value_from_tree(self, value, iEvent=None, iEntry=None):
        if value not in self.variables[self.input_type]:
            return None
        
        value = self.variables[self.input_type][value]
        
        if value in self.tree.keys():
            if iEntry is None:
                if iEvent is None:
                    return self.tree[value].array()
                else:
                    return self.tree[value].array()[iEvent]
            else:
                return self.tree[value].array()[iEvent][iEntry]
        
        return None

    def get_array_n_dimensions(self, value):
        if value not in self.variables[self.input_type]:
            return None
    
        value = self.variables[self.input_type][value]
    
        if value in self.tree.keys():
            return self.tree[value].array().ndim
    
        return None
