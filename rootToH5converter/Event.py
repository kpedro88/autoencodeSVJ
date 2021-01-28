import numpy as np
from Jet import Jet
from PhysObject import PhysObject


class Event:
    def __init__(self, data_processor, i_event, delta_r):
        """
        Reads/calculates event level features, loads jets, tracks, photons and neutral hadrons.
        Adds jet constituents to jets.
        """
    
        self.i_event = i_event
        self.delta_r = delta_r
        self.data_processor = data_processor
    
        # account for the fact that in Delphes MET is stored in an array with just one element
        i_object = None
        if data_processor.get_array_n_dimensions("MET_pt") == 2:
            i_object = 0
        
        # read event features from tree
        self.metPt = data_processor.get_value_from_tree("MET_pt", i_event, i_object)
        self.metPhi = data_processor.get_value_from_tree("MET_phi", i_event, i_object)
        self.metEta = data_processor.get_value_from_tree("MET_eta", i_event, i_object)
        
        self.nJets = data_processor.get_value_from_tree("N_jets", i_event)
        self.nTracks = data_processor.get_value_from_tree("N_tracks", i_event)
        self.nNeutralHadrons = data_processor.get_value_from_tree("N_neutral_hadrons", i_event)
        self.nPhotons = data_processor.get_value_from_tree("N_photons", i_event)
        
        # load tracks from tree
        self.tracks = []
        self.fill_tracks()
        
        # load neutral hadrons from tree
        self.neutral_hadrons = []
        self.fill_neutral_hadrons()

        # load photons from tree
        self.photons = []
        self.fill_photons()
        
        # load jets from tree
        self.jets = []
        self.fill_jets()

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
    
    def fill_tracks(self):
        if self.nTracks is None:
            return
            
        for i_track in range(0, self.nTracks):
            track = PhysObject(eta=self.data_processor.get_value_from_tree("Track_eta", self.i_event, i_track),
                               phi=self.data_processor.get_value_from_tree("Track_phi", self.i_event, i_track),
                               pt=self.data_processor.get_value_from_tree("Track_pt", self.i_event, i_track),
                               mass=self.data_processor.get_value_from_tree("Track_mass", self.i_event, i_track))
            self.tracks.append(track)
    
    def fill_neutral_hadrons(self):
        if self.nNeutralHadrons is None:
            return
        
        for i_neutral in range(0, self.nNeutralHadrons):
            neutral = PhysObject(eta=self.data_processor.get_value_from_tree("Neutral_eta", self.i_event, i_neutral),
                                 phi=self.data_processor.get_value_from_tree("Neutral_phi", self.i_event, i_neutral),
                                 pt=self.data_processor.get_value_from_tree("Neutral_pt", self.i_event, i_neutral),
                                 mass=self.data_processor.get_value_from_tree("Neutral_mass", self.i_event, i_neutral))
            self.neutral_hadrons.append(neutral)
    
    def fill_photons(self):
        if self.nPhotons is None:
            return
            
        for i_photon in range(0, self.nPhotons):
            photon = PhysObject(eta=self.data_processor.get_value_from_tree("Photon_eta", self.i_event, i_photon),
                                phi=self.data_processor.get_value_from_tree("Photon_phi", self.i_event, i_photon),
                                pt=self.data_processor.get_value_from_tree("Photon_pt", self.i_event, i_photon),
                                mass=self.data_processor.get_value_from_tree("Photon_mass", self.i_event, i_photon))
            self.photons.append(photon)
    
    def fill_jets(self):
        if self.nJets is None:
            return
            
        for i_jet in range(0, self.nJets):
            jet = Jet(eta=self.data_processor.get_value_from_tree("Jet_eta", self.i_event, i_jet),
                      phi=self.data_processor.get_value_from_tree("Jet_phi", self.i_event, i_jet),
                      pt=self.data_processor.get_value_from_tree("Jet_pt", self.i_event, i_jet),
                      mass=self.data_processor.get_value_from_tree("Jet_mass", self.i_event, i_jet),
                      flavor=self.data_processor.get_value_from_tree("Jet_flavor", self.i_event, i_jet),
                      n_charged=self.data_processor.get_value_from_tree("Jet_nCharged", self.i_event, i_jet),
                      n_neutral=self.data_processor.get_value_from_tree("Jet_nNeutral", self.i_event, i_jet),
                      ch_hef=self.data_processor.get_value_from_tree("Jet_chHEF", self.i_event, i_jet),
                      ne_hef=self.data_processor.get_value_from_tree("Jet_neHEF", self.i_event, i_jet))
            
            # check if tree contains links between tracks and jets
            jet_index = -1
            track_jet_index = None
            if self.data_processor.get_value_from_tree("Track_jet_index") is not None:
                track_jet_index = self.data_processor.get_value_from_tree("Track_jet_index", self.i_event)
                jet_index = i_jet
            
            # fill jet constituents
            jet.fill_constituents(self.tracks, self.neutral_hadrons, self.photons, self.delta_r, jet_index, track_jet_index)
            
            self.jets.append(jet)
    
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
        
        met_py = self.metPt * np.sin(self.metPhi)
        met_px = self.metPt * np.cos(self.metPhi)
    
        Mjj = dijet_vector.M()
        Mjj2 = Mjj * Mjj
        ptjj = dijet_vector.Pt()
        ptjj2 = ptjj * ptjj
        ptMet = dijet_vector.Px() * met_px + dijet_vector.Py() * met_py
    
        MT = np.sqrt(Mjj2 + 2. * (np.sqrt(Mjj2 + ptjj2) * self.metPt - ptMet))
        
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
            self.metPt,
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
