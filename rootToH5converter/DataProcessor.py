from enum import Enum


class InputTypes(Enum):
    Delphes = 0,
    nanoAOD = 1,
    PFnanoAOD = 2


class DataProcessor:
    def __init__(self, tree, input_type):
        """
        Creates DataProcessor objects which knows names of branches for different input types.
        Pre-loads all branches for later use.
        """
        
        if input_type not in InputTypes:
            print("\n\nERROR -- DataProcessor: unknown input type: ", input_type, "\n\n")
            exit(0)
        
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
        
        # pre-load all branches for this tree to avoid calling this for every event/track/jet
        self.branches = {}
        for key, value in self.variables[input_type].items():
            if value in tree.keys():
                self.branches[key] = tree[value].array()
        
    def get_value_from_tree(self, variable, i_event=None, i_entry=None):
        """
        Returns value of given variable for given event. If event contains an array of such variable (e.g. jets pt),
        i_entry must be also specified.
        """
        if variable not in self.branches.keys():
            return None
        
        if i_entry is None:
            return self.branches[variable][i_event]
        else:
            return self.branches[variable][i_event][i_entry]
        
    def get_array_n_dimensions(self, variable):
        """
        Returns number of dimensions of the tree leaf for given variable (1 is a number, 2 is a vector etc.)
        """
        
        if variable not in self.branches.keys():
            return None
    
        return self.branches[variable].ndim
