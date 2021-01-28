from InputTypes import *

class ConstituentBranches:
    
    def __init__(self, data_processor):
        self.track_eta = data_processor.get_value_from_tree("Track_eta")
        self.track_phi = data_processor.get_value_from_tree("Track_phi")
        self.track_pt = data_processor.get_value_from_tree("Track_pt")
        self.track_mass = data_processor.get_value_from_tree("Track_mass")
        self.track_jet_index = data_processor.get_value_from_tree("Track_jet_index")
    
        self.neutral_hadron_eta = data_processor.get_value_from_tree("Neutral_eta")
        self.neutral_hadron_phi = data_processor.get_value_from_tree("Neutral_phi")
        self.neutral_hadron_pt = data_processor.get_value_from_tree("Neutral_pt")
        self.neutral_hadron_mass = data_processor.get_value_from_tree("Neutral_mass")
    
        self.photon_eta = data_processor.get_value_from_tree("Photon_eta")
        self.photon_phi = data_processor.get_value_from_tree("Photon_phi")
        self.photon_pt = data_processor.get_value_from_tree("Photon_pt")
        self.photon_mass = data_processor.get_value_from_tree("Photon_mass")