import uproot
import numpy as np
import energyflow as ef
import os
import sys
import h5py
from Jet import Jet
from Event import Event
import awkward as ak

class Converter:

    def __init__(
        self,
        input_paths,
        output_path,
        output_file_prefix,
        jet_delta_r=0.5,
        max_n_constituents=100,
        save_constituents=False,
        energyflow_basis_degree=-1,
    ):
        self.output_path = output_path
        self.output_file_prefix = output_file_prefix

        self.set_input_paths_and_selections(input_paths=input_paths)

        # core tree, add files, add all trees
        self.files = {path: uproot.open(path) for path in self.input_file_paths}

        self.trees = {}
        self.input_types = {}
        
        for path, file in self.files.items():
            
            for key in file.keys():
                if key.startswith("Delphes"):
                    self.trees[path] = file["Delphes"]
                    self.input_types[path] = "Delphes"
                    print("Adding Delphes tree")
                elif key.startswith("Events"):
                    self.trees[path] = file[key]
                    self.input_types[path] = "nanoAOD"
                    print("Adding nanoAOD tree: ", key)
                else:
                    print("Unknown tree type: ", key, ". Skipping...")
        
        
        print("Loaded trees: ", self.trees)
        print("Corresponding types: ", self.input_types)
        
        self.sizes = [tree.num_entries for tree in self.trees.values()]
        self.nEvents = sum(self.sizes)

        print("Found {0} files".format(len(self.files)))
        print("Found {0} delphes trees".format(len(self.trees)))
        print("Found {0} total events".format(self.nEvents))

        
        self.jet_delta_r = jet_delta_r
        self.n_jets = 2

        self.max_n_constituents = max_n_constituents
        self.event_features = None
        self.jet_features = None
        self.jet_constituents = None
        self.energy_flow_bases = None
        self.save_constituents = save_constituents
        self.efbn = energyflow_basis_degree

        if self.efbn < 0:
            self.save_eflow = False
            self.efbn = 0
            self.efp_size = 0
        else:
            self.save_eflow = True
            print("creating energyflow particle set with degree d <= {0}...".format(self.efbn))
            self.efpset = ef.EFPSet("d<={0}".format(self.efbn), measure='hadr', beta=1.0, normed=True, verbose=True)
            self.efp_size = self.efpset.count()
            print("eflow set is size {}".format(self.efp_size))
        
        print("found {0} selected events, out of a total of {1}".format(sum(map(len, list(self.selections.values()))), self.nEvents))

    def set_input_paths_and_selections(self, input_paths):
        self.selections = {}
        self.input_file_paths = []
        
        for path in input_paths:
            with open(path, 'r') as file:
                line = [lines.strip('\n') for lines in file.readlines()]
            for elements in line:
                file_name, selections = elements.split(': ')
                self.input_file_paths.append(file_name)
                self.selections[file_name] = list(map(int, selections.split()))


    def convert(self):
        
        total_size = sum(map(len, list(self.selections.values()))) + 1
        total_count = 0

        self.event_features = np.empty((total_size, len(Event.get_features_names())))
        print("event feature shapes: {}".format(self.event_features.shape))

        self.jet_features = np.empty((total_size, self.n_jets, len(Jet.get_feature_names())))
        print("jet feature shapes: {}".format(self.event_features.shape))

        self.jet_constituents = np.empty(
            (total_size, self.n_jets, self.max_n_constituents, len(Jet.get_constituent_feature_names())))
        print("jet constituent shapes: {}".format(self.jet_constituents.shape))

        self.energy_flow_bases = np.empty((total_size, self.n_jets, self.efp_size))
        print("eflow bases shapes: {}".format(self.energy_flow_bases.shape))

        if not self.save_constituents:
            print("ignoring jet constituents")
    
    
        for file_name, tree in self.trees.items():
            print("Loading events from file: ", file_name)
            input_type = self.input_types[file_name]
            print("Input type was recognised to be ", input_type)

            track_eta = []
            track_phi = []
            track_pt = []
            track_mass = []

            neutral_hadron_eta = []
            neutral_hadron_phi = []
            neutral_hadron_pt = []
            neutral_hadron_mass = []

            photon_eta = []
            photon_phi = []
            photon_pt = []
            photon_mass = []

            if input_type == "Delphes":
                track_eta = tree["EFlowTrack/EFlowTrack.Eta"].array()
                track_phi = tree["EFlowTrack/EFlowTrack.Phi"].array()
                track_pt = tree["EFlowTrack/EFlowTrack.PT"].array()
                track_mass = 0
                
                neutral_hadron_eta = tree["EFlowNeutralHadron/EFlowNeutralHadron.Eta"].array()
                neutral_hadron_phi = tree["EFlowNeutralHadron/EFlowNeutralHadron.Phi"].array()
                neutral_hadron_pt = tree["EFlowNeutralHadron/EFlowNeutralHadron.ET"].array()
                neutral_hadron_mass = 0
            
                photon_eta = tree["Photon/Photon.Eta"].array()
                photon_phi = tree["Photon/Photon.Phi"].array()
                photon_pt = tree["Photon/Photon.PT"].array()
                photon_mass = 0
                
            elif input_type == "nanoAOD":
                print("WARNING -- handling of tracks for nanoAOD not implemented!!!")
                print("WARNING -- handling of neutral hadrons for nanoAOD not implemented!!!")

                photon_eta = tree["Photon_eta"].array()
                photon_phi = tree["Photon_phi"].array()
                photon_pt = tree["Photon_pt"].array()
                photon_mass = tree["Photon_mass"].array()

            print("\n\nTrack array type: ", type(track_pt), "\n\n")

            for iEvent in self.selections[file_name]:
    
                
                event = Event(tree, input_type, iEvent,
                              track_eta, track_phi, track_pt, track_mass,
                              neutral_hadron_eta, neutral_hadron_phi, neutral_hadron_pt, neutral_hadron_mass,
                              photon_eta, photon_phi, photon_pt, photon_mass, self.jet_delta_r
                              )
                
                
                if event.nJets < 2:
                    print("WARNING -- event has less than 2 jets! Skipping...")
                    continue

                event.calculate_internals()
                
                print("Event: ", iEvent)
                # event.print()
                
                self.event_features[total_count, :] = np.asarray(event.get_features())

                for iJet, jet in enumerate(event.jets):
                    if iJet == 2:
                        break

                    self.jet_features[total_count, iJet, :] = event.jets[iJet].get_features()

                    if self.save_constituents:
                        self.jet_constituents[total_count, iJet, :] = self.get_jet_constituents(jet.constituents)

                    if self.save_eflow:
                        self.energy_flow_bases[total_count, iJet, :] = jet.get_EFPs(self.efpset)
                
                total_count += 1

        # remove redundant rows for events that didn't meet some criteria
        for i in range(0, total_size-total_count):
            self.event_features = np.delete(self.event_features, -1, axis=0)
            self.jet_features = np.delete(self.jet_features, -1, axis=0)

    def pad_to_n(self, data, n, sort_index):
        data = data[np.argsort(data[:,sort_index]),:][::-1][:n,:]
        data = np.pad(data, ((0,n-data.shape[0]), (0,0)), 'constant')
        return data

    def get_jet_constituents(self, constituentp4s):
    
        ret = -np.ones((len(constituentp4s), len(Jet.get_constituent_feature_names())))
        for i, c in enumerate(constituentp4s):
            ret[i, :] = [c.Eta(), c.Phi(), c.Pt(), c.Rapidity(), c.E()]
    
        return self.pad_to_n(ret, self.max_n_constituents, 2)

    def save(self, output_file_name=None):
    
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        output_file_name = output_file_name or os.path.join(self.output_path, "{}_data.h5".format(self.output_file_prefix))
    
        if not output_file_name.endswith(".h5"):
            output_file_name += ".h5"
    
        print("saving h5 data to file {0}".format(output_file_name))
    
        f = h5py.File(output_file_name, "w")
        features = f.create_group("event_features")
        assert self.event_features.shape[-1] == len(Event.get_features_names())
        print("creating feature 'event_features'")
        features.create_dataset('data', data=self.event_features)
        features.create_dataset('labels', data=Event.get_features_names())
    
        jet_features = f.create_group("jet_features")
        # assert self.jet_features.shape[-1] == len(Jet.get_feature_names())
        print("creating feature 'jet_features'")
        jet_features.create_dataset('data', data=self.jet_features)
        jet_features.create_dataset('labels', data=Jet.get_feature_names())
    
        if self.save_constituents:
            jet_constituents = f.create_group("jet_constituents")
            assert self.jet_constituents.shape[-2] == self.max_n_constituents
            assert self.jet_constituents.shape[-1] == len(Jet.get_constituent_feature_names())
            print("creating feature 'jet_constituents'")
            jet_constituents.create_dataset('data', data=self.jet_constituents)
            jet_constituents.create_dataset('labels', data=Jet.get_constituent_feature_names())
    
        if self.save_eflow:
            eflow = f.create_group("jet_eflow_variables")
            assert self.energy_flow_bases.shape[-1] == self.efp_size
            print("creating feature 'jet_eflow_variables'")
            eflow.create_dataset('data', data=self.energy_flow_bases)
            eflow.create_dataset('labels', data=[str(i) for i in range(self.efp_size)])
    
        print("Successfully saved!")
        f.close()