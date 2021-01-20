import uproot
import numpy as np
import energyflow as ef
import os
import sys
import h5py
from Jet import Jet
from Event import Event

class Converter:

    def __init__(
        self,
        input_paths,
        output_path,
        output_file_prefix,
        jetDR=0.5,
        n_constituent_particles=100,
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

        
        self.jetDR = jetDR
        self.n_jets = 2

        self.jet_feature_names = [
            'Eta',
            'Phi',
            'Pt',
            'M',
            'ChargedFraction',
            'PTD',
            'Axis2',
            'Flavor',
            'Energy',
        ]

        self.jet_constituent_names = [
            'Eta',
            'Phi',
            'PT',
            'Rapidity',
            'Energy',
        ]

        self.n_constituent_particles=n_constituent_particles
        self.event_features = None
        self.jet_features = None
        self.jet_constituents = None
        self.energy_flow_bases = None

        hlf_dict = {}
        particle_dict = {}

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
        # self.selections_abs = np.asarray([sum(self.sizes[:s[0]]) + s[1] for s in self.selections])
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


    def convert(self, rng=(-1, -1)):
        rng = list(rng)

        gmin, gmax = min(self.sizes), max(self.sizes)

        if rng[0] < 0 or rng[0] > gmax:
            rng[0] = 0

        if rng[1] > gmax or rng[1] < 0:
            rng[1] = gmax

        nmin, nmax = rng
        selections_iter = self.selections.copy()

        for k, v in list(selections_iter.items()):
            v = np.asarray(v).astype(int)
            selections_iter[k] = v[(v > nmin) & (v < nmax)]

        total_size = sum(map(len, list(selections_iter.values()))) + 1
        total_count = 0

        print("selecting on range {0}".format(rng))
        self.event_features = np.empty((total_size, len(Event.get_features_names())))
        print("event feature shapes: {}".format(self.event_features.shape))

        self.jet_features = np.empty((total_size, self.n_jets, len(self.jet_feature_names)))
        print("jet feature shapes: {}".format(self.event_features.shape))

        self.jet_constituents = np.empty(
            (total_size, self.n_jets, self.n_constituent_particles, len(self.jet_constituent_names)))
        print("jet constituent shapes: {}".format(self.jet_constituents.shape))

        self.energy_flow_bases = np.empty((total_size, self.n_jets, self.efp_size))
        print("eflow bases shapes: {}".format(self.energy_flow_bases.shape))

        if not self.save_constituents:
            print("ignoring jet constituents")
    
    
        for file_name, tree in self.trees.items():
            print("Loading events from file: ", file_name)
            input_type = self.input_types[file_name]
            print("Input type was recognised to be ", input_type)
    
    
            for iEvent in self.selections[file_name]:
                
                nJets = 0
                
                if input_type == "Delphes":
                    nJets = tree["Jet_size"].array()[iEvent]
                elif input_type == "nanoAOD":
                    nJets = tree["nJet"].array()[iEvent]
                else:
                    print("\n\nERROR -- unknown input type!\n\n")
                    exit(0)
                
                if nJets < 2:
                    print("WARNING -- event has less than 2 jets! Skipping...")
                    continue
                
                jets = []
                
                for iJet in range(0, nJets):
                    jet = Jet()

                    if input_type == "Delphes":
                        jet.eta = tree["Jet/Jet.Eta"].array()[iEvent][iJet]
                        jet.phi = tree["Jet/Jet.Phi"].array()[iEvent][iJet]
                        jet.pt = tree["Jet/Jet.PT"].array()[iEvent][iJet]
                        jet.mass = tree["Jet/Jet.Mass"].array()[iEvent][iJet]
                        jet.nCharged = tree["Jet/Jet.NCharged"].array()[iEvent][iJet]
                        jet.nNeutral = tree["Jet/Jet.NNeutrals"].array()[iEvent][iJet]
                    elif input_type == "nanoAOD":
                        jet.eta = tree["Jet_eta"].array()[iEvent][iJet]
                        jet.phi = tree["Jet_phi"].array()[iEvent][iJet]
                        jet.pt = tree["Jet_pt"].array()[iEvent][iJet]
                        jet.mass = tree["Jet_mass"].array()[iEvent][iJet]
                      
                        jet.chargedHadronEnergyFraction = tree["Jet_chHEF"].array()[iEvent][iJet]
                        jet.chargedHadronEnergyFraction = tree["Jet_neHEF"].array()[iEvent][iJet]
                    
                
                    jets.append(jet)
                
                print("Event: ", iEvent, "\tjets: ")
                for jet in jets:
                    jet.print()

                # constituents_by_jet = self.get_constituent_p4s(tree, jets, self.jetDR)

                event = Event()
                event.jets = jets

                if input_type == "Delphes":
                    event.met = tree["MissingET.MET"].array()[iEvent][0]
                    event.metPhi = tree["MissingET.Phi"].array()[iEvent][0]
                    event.metEta = tree["MissingET.Eta"].array()[iEvent][0]
                elif input_type == "nanoAOD":
                    event.met = tree["MET_pt"].array()[iEvent]
                    event.metPhi = tree["MET_phi"].array()[iEvent]
                    event.metEta = 0

                
                
                event.calculate_internals()

                event_features = event.get_features()

                print("Event features: ", event_features)
                print("Event features (asarray): ", np.asarray(event_features))
                self.event_features[total_count, :] = np.asarray(event_features)
                
                
                print("\n\nCurrent event features: ", self.event_features, "\n\n")

                

                total_count += 1

        
        for i in range(0, total_size-total_count):
            self.event_features = np.delete(self.event_features, -1, axis=0)
        
        
        
        
        # selection is implicit: looping only through total selectinos
        # for tree_n, tree_name in enumerate(self.input_file_paths):
        #
        #     for event_n, event_index in enumerate(selections_iter[tree_name]):
        #
        #         print(
        #             'tree {0}, event {1}, index {2}, total count {3}'.format(tree_n, event_n, event_index, total_count))
        #
        #         # tree (and, get the entry)
        #         tree = self.trees[tree_n]
        #         tree.GetEntry(event_index)
        #
        #         # jets
        #         jets_raw = [tree.Jet[i] for i in range(min([self.n_jets, tree.Jet_size]))]
        #
        #         jets = [j.P4() for j in jets_raw]
        #
        #         # constituent 4-vectors per jet
        #         constituents_by_jet = self.get_constituent_p4s(tree, jets, self.jetDR)
        #
        #         self.event_features[total_count, :] = np.asarray(self.get_event_features(tree))
        #
        #         for jet_n, (jet_raw, jet_p4, constituents) in enumerate(zip(jets_raw, jets, constituents_by_jet)):
        #
        #             self.jet_features[total_count, jet_n, :] = self.get_jet_features(jet_raw, constituents)
        #
        #             if self.save_constituents:
        #                 self.jet_constituents[total_count, jet_n, :] = self.get_jet_constituents(constituents)
        #
        #             if self.save_eflow:
        #                 self.energy_flow_bases[total_count, jet_n, :] = self.get_eflow_variables(constituents)

        #         total_count += 1
    
        return None

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
    
        # jet_features = f.create_group("jet_features")
        # assert self.jet_features.shape[-1] == len(self.jet_feature_names)
        # print("creating feature 'jet_features'")
        # jet_features.create_dataset('data', data=self.jet_features)
        # jet_features.create_dataset('labels', data=self.jet_feature_names)
    
        # if self.save_constituents:
        #     jet_constituents = f.create_group("jet_constituents")
        #     assert self.jet_constituents.shape[-2] == self.n_constituent_particles
        #     assert self.jet_constituents.shape[-1] == len(self.jet_constituent_names)
        #     print("creating feature 'jet_constituents'")
        #     jet_constituents.create_dataset('data', data=self.jet_constituents)
        #     jet_constituents.create_dataset('labels', data=self.jet_constituent_names)
    
        # if self.save_eflow:
        #     eflow = f.create_group("jet_eflow_variables")
        #     assert self.energy_flow_bases.shape[-1] == self.efp_size
        #     print("creating feature 'jet_eflow_variables'")
        #     eflow.create_dataset('data', data=self.energy_flow_bases)
        #     eflow.create_dataset('labels', data=[str(i) for i in range(self.efp_size)])
    
        print("Successfully saved!")
        f.close()