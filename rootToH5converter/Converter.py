import uproot
import numpy as np
import energyflow as ef
import os
import h5py
from Jet import Jet
from Event import Event

from ConstituentBranches import ConstituentBranches

class Converter:

    def __init__(self, input_path, store_n_jets, jet_delta_r, max_n_constituents, EFP_degree):
        self.set_input_paths_and_selections(input_path=input_path)

        # read files, trees and recognize input type
        self.files = {path: uproot.open(path) for path in self.input_file_paths}
        self.trees = {}
        self.input_types = {}
        self.read_trees()
        self.n_all_events = sum([tree.num_entries for tree in self.trees.values()])
        self.n_events = sum(map(len, list(self.selections.values()))) + 1

        print("Found {0} file(s)".format(len(self.files)))
        print("Found {0} tree(s)".format(len(self.trees)))
        print("Found ", self.n_events - 1, " selected events, out of a total of ", self.n_all_events)

        # set internal parameters
        self.jet_delta_r = jet_delta_r
        self.max_n_constituents = max_n_constituents
        self.save_constituents = False if max_n_constituents < 0 else True
        self.n_jets = store_n_jets
        self.save_EFPs = False if EFP_degree < 0 else True
        self.EFP_size = 0

        # initialize EFP set
        if EFP_degree >= 0:
            print("\n\n=======================================================")
            print("Creating energyflow particle set with degree d <= {0}...".format(EFP_degree))
            self.efpset = ef.EFPSet("d<={0}".format(EFP_degree), measure='hadr', beta=1.0, normed=True, verbose=True)
            self.EFP_size = self.efpset.count()
            print("EFP set is size: {}".format(self.EFP_size))
            print("=======================================================\n\n")
            
        # prepare arrays for event & jet features, EFPs and jet constituents
        self.event_features = np.empty((self.n_events, len(Event.get_features_names())))
        self.jet_features = np.empty((self.n_events, self.n_jets, len(Jet.get_feature_names())))
        self.jet_constituents = np.empty((self.n_events, self.n_jets, self.max_n_constituents, len(Jet.get_constituent_feature_names())))
        self.energy_flow_bases = np.empty((self.n_events, self.n_jets, self.EFP_size))
        
    def set_input_paths_and_selections(self, input_path):
        """
        Reads input file with paths to ROOT files and corresponding event selections.
        """
        self.selections = {}
        self.input_file_paths = []
        
        with open(input_path, 'r') as file:
            line = [lines.strip('\n') for lines in file.readlines()]
        for elements in line:
            file_name, selections = elements.split(': ')
            self.input_file_paths.append(file_name)
            self.selections[file_name] = list(map(int, selections.split()))

    def read_trees(self):
        """
        Reads input ROOT files, extracts trees and recognizes type of the input file (Delphes/nanoAOD/PFnanoAOD).
        """
        for path, file in self.files.items():
        
            for key in file.keys():
                if key.startswith("Delphes"):
                    self.trees[path] = file["Delphes"]
                    self.input_types[path] = "Delphes"
                    print("Adding Delphes tree")
                elif key.startswith("Events"):
                    self.trees[path] = file[key]
                
                    if file[key]["JetPFCands_eta"] is not None:
                        self.input_types[path] = "PFnanoAOD"
                    else:
                        self.input_types[path] = "nanoAOD"
                
                    print("Adding nanoAOD tree: ", key)
                else:
                    print("Unknown tree type: ", key, ". Skipping...")

    def convert(self):
        
        total_count = 0
        
        for file_name, tree in self.trees.items():
    
            print("\n\n=======================================================")
            print("Loading events from file: ", file_name)
            input_type = self.input_types[file_name]
            print("Input type was recognised to be: ", input_type)

            constituentBranches = ConstituentBranches(tree, input_type)

           
                

            for iEvent in self.selections[file_name]:
                print("\n\n------------------------------")
                print("Event: ", iEvent)
                
                event = Event(tree, input_type, iEvent, constituentBranches, self.jet_delta_r)
                
                
                if event.nJets < 2:
                    print("WARNING -- event has less than 2 jets! Skipping...")
                    print("------------------------------\n\n")
                    continue

                event.calculate_internals()
                event.print()

                hasJetWithNoConstituents = False
                
                for i in range(0, self.n_jets):
                    if len(event.jets[i].constituents) == 0:
                        hasJetWithNoConstituents = True
                        break
                
                if hasJetWithNoConstituents:
                    print("WARNING -- one of the jets that should be stored has no constituents. Skipping...")
                    continue
                
                
                self.event_features[total_count, :] = np.asarray(event.get_features())

                for iJet, jet in enumerate(event.jets):
                    if iJet == self.n_jets:
                        break

                    self.jet_features[total_count, iJet, :] = event.jets[iJet].get_features()

                    if self.save_constituents:
                        self.jet_constituents[total_count, iJet, :] = self.get_jet_constituents(jet.constituents)

                    if self.save_EFPs:
                        self.energy_flow_bases[total_count, iJet, :] = jet.get_EFPs(self.efpset)
                        
                
                total_count += 1
                print("------------------------------\n\n")

            print("\n\n=======================================================")

        # remove redundant rows for events that didn't meet some criteria
        for i in range(0, self.n_events - total_count):
            self.event_features = np.delete(self.event_features, -1, axis=0)
            self.jet_features = np.delete(self.jet_features, -1, axis=0)
            self.energy_flow_bases = np.delete(self.energy_flow_bases, -1, axis=0)
            self.jet_constituents = np.delete(self.jet_constituents, -1, axis=0)

    def pad_to_n(self, data, n, sort_index):
        data = data[np.argsort(data[:,sort_index]),:][::-1][:n,:]
        data = np.pad(data, ((0,n-data.shape[0]), (0,0)), 'constant')
        return data

    def get_jet_constituents(self, constituentp4s):
    
        ret = -np.ones((len(constituentp4s), len(Jet.get_constituent_feature_names())))
        for i, c in enumerate(constituentp4s):
            ret[i, :] = [c.Eta(), c.Phi(), c.Pt(), c.Rapidity(), c.E()]
    
        return self.pad_to_n(ret, self.max_n_constituents, 2)

    def save(self, output_file_name):
    
        path_directory = os.path.dirname(output_file_name)
    
        if not os.path.exists(path_directory) and path_directory is not None and path_directory != '':
            os.mkdir(path_directory)
        
        if not output_file_name.endswith(".h5"):
            output_file_name += ".h5"

        print("\n\n=======================================================")
        print("Saving h5 data to file: ", output_file_name)
    
        f = h5py.File(output_file_name, "w")
        features = f.create_group("event_features")
        features.create_dataset('data', data=self.event_features)
        features.create_dataset('labels', data=Event.get_features_names())
    
        jet_features = f.create_group("jet_features")
        jet_features.create_dataset('data', data=self.jet_features)
        jet_features.create_dataset('labels', data=Jet.get_feature_names())
    
        if self.save_constituents:
            jet_constituents = f.create_group("jet_constituents")
            jet_constituents.create_dataset('data', data=self.jet_constituents)
            jet_constituents.create_dataset('labels', data=Jet.get_constituent_feature_names())
    
        if self.save_EFPs:
            eflow = f.create_group("jet_eflow_variables")
            eflow.create_dataset('data', data=self.energy_flow_bases)
            eflow.create_dataset('labels', data=[str(i) for i in range(self.EFP_size)])

        f.close()
        print("Successfully saved!")
        print("=======================================================\n\n")
        