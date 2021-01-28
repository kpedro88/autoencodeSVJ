import numpy as np
from Jet import Jet
from PhysObject import PhysObject


class Event:
    def __init__(self, data_processor, i_event, delta_r):
        """
        Reads/calculates event level features, loads jets, tracks, photons and neutral hadrons.
        Adds jet constituents to jets.
        """
    
        # account for the fact that in Delphes MET is stored in an array with just one element
        i_object = None
        if data_processor.get_array_n_dimensions("MET_pt") == 2:
            i_object = 0
        
        # read event features from tree
        self.met    = data_processor.get_value_from_tree("MET_pt", i_event, i_object)
        self.metPhi = data_processor.get_value_from_tree("MET_phi", i_event, i_object)
        self.metEta = data_processor.get_value_from_tree("MET_eta", i_event, i_object)
        
        self.nJets           = data_processor.get_value_from_tree("N_jets", i_event)
        self.nTracks         = data_processor.get_value_from_tree("N_tracks", i_event)
        self.nNeutralHadrons = data_processor.get_value_from_tree("N_neutral_hadrons", i_event)
        self.nPhotons        = data_processor.get_value_from_tree("N_photons", i_event)
        
        # load tracks from tree
        self.tracks = []
        if self.nTracks is not None:
            for i_track in range(0, self.nTracks):
                track = PhysObject(eta  = data_processor.get_value_from_tree("Track_eta" , i_event, i_track),
                                   phi  = data_processor.get_value_from_tree("Track_phi" , i_event, i_track),
                                   pt   = data_processor.get_value_from_tree("Track_pt"  , i_event, i_track),
                                   mass = data_processor.get_value_from_tree("Track_mass", i_event, i_track))
                self.tracks.append(track)
        
        # load neutral hadrons from tree
        self.neutral_hadrons = []
        if self.nNeutralHadrons is not None:
            for i_neutral in range(0, self.nNeutralHadrons):
                neutral = PhysObject(eta  = data_processor.get_value_from_tree("Neutral_eta" ,i_event, i_neutral),
                                     phi  = data_processor.get_value_from_tree("Neutral_phi" ,i_event, i_neutral),
                                     pt   = data_processor.get_value_from_tree("Neutral_pt"  ,i_event, i_neutral),
                                     mass = data_processor.get_value_from_tree("Neutral_mass",i_event, i_neutral))
                self.neutral_hadrons.append(neutral)

        # load photons from tree
        self.photons = []
        if self.nPhotons is not None:
            for i_photon in range(0, self.nPhotons):
                photon = PhysObject(eta  = data_processor.get_value_from_tree("Photon_eta" , i_event, i_photon),
                                    phi  = data_processor.get_value_from_tree("Photon_phi" , i_event, i_photon),
                                    pt   = data_processor.get_value_from_tree("Photon_pt"  , i_event, i_photon),
                                    mass = data_processor.get_value_from_tree("Photon_mass", i_event, i_photon))
                self.photons.append(photon)
        
        # load jets from tree
        self.jets = []
        if self.nJets is not None:
            for iJet in range(0, self.nJets):
                self.jets.append(Jet(data_processor, i_event, iJet))
    
        # fill jet constituents
        for iJet, jet in enumerate(self.jets):
            
            # check if tree contains links between tracks and jets
            jet_index = -1
            track_jet_index = None
            if data_processor.get_value_from_tree("Track_jet_index") is not None:
                track_jet_index = data_processor.get_value_from_tree("Track_jet_index", i_event)
                jet_index = iJet
            
            jet.fill_constituents(self.tracks, self.neutral_hadrons, self.photons, delta_r, jet_index, track_jet_index)

        # calculate remaining event features
        self.Mjj = None
        self.MT = None
    
        if self.nJets >= 2:
            self.calculate_internals()
    
    def print(self):
        """
        Prints basic informations about the event.
        """
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
        """
        Checks if event contains jets with not constituents. Analyzes only first `max_n_jets` highest pt jets.
        """
        
        has_jets_with_no_constituents = False
    
        for i in range(0, max_n_jets):
            if len(self.jets[i].constituents) == 0:
                has_jets_with_no_constituents = True
                break
                
        return has_jets_with_no_constituents
    
    def calculate_internals(self):
        """
        Calculates Mjj and MT for the two leading jets.
        """
        
        if len(self.jets) <= 1:
            print("ERROR -- events has less than 2 jets, which should never happen!")
            return
        
        if len(self.jets) != 2:
            print("WARNING -- expected two jets in the event, but there are ", len(self.jets))
        
        dijet_vector = self.jets[0].get_four_vector() + self.jets[1].get_four_vector()
        
        print("\n\n MET phi: ", self.metPhi)
        
        met_py = self.met * np.sin(self.metPhi)
        met_px = self.met * np.cos(self.metPhi)
    
        Mjj = dijet_vector.M()
        Mjj2 = Mjj * Mjj
        ptjj = dijet_vector.Pt()
        ptjj2 = ptjj * ptjj
        ptMet = dijet_vector.Px() * met_px + dijet_vector.Py() * met_py
    
        MT = np.sqrt(Mjj2 + 2. * (np.sqrt(Mjj2 + ptjj2) * self.met - ptMet))
        
        self.Mjj = Mjj
        self.MT = MT

    def are_jets_ordered_by_pt(self):
        """
        Checks if all jets in the event are ordered by pt.
        """
        for i_jet in range(1, self.nJets):
            if self.jets[i_jet].get_four_vector().Pt() > self.jets[i_jet-1].get_four_vector().Pt():
                return False
            
        return True

    def get_features(self):
        """
        Returns event features.
        """
        return [
            self.met,
            self.metEta,
            self.metPhi,
            self.MT,
            self.Mjj,
        ]
    
    @staticmethod
    def get_features_names():
        """
        Returns names of event features.
        """
        return [
            'MET',
            'METEta',
            'METPhi',
            'MT',
            'Mjj',
        ]
