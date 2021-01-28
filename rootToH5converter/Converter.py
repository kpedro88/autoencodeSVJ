import uproot
import numpy as np
import energyflow as ef
import os
import h5py
from Jet import Jet
from Event import Event
from DataProcessor import *
from enum import Enum


class OutputTypes(Enum):
    EventFeatures = 0,
    JetFeatures = 1,
    EPFs = 2,
    JetConstituents = 3
    

class Converter:

    def __init__(self, input_path, store_n_jets, jet_delta_r, max_n_constituents, efp_degree):
        """
        Reads input trees, recognizes input types, initializes EFP processor and prepares all arrays needed to
        store output variables.
        """
        
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
        self.max_n_jets = store_n_jets
        self.EFP_size = 0

        # initialize EFP set
        if efp_degree >= 0:
            print("\n\n=======================================================")
            print("Creating energyflow particle set with degree d <= {0}...".format(efp_degree))
            self.efpset = ef.EFPSet("d<={0}".format(efp_degree), measure='hadr', beta=1.0, normed=True, verbose=True)
            self.EFP_size = self.efpset.count()
            print("EFP set is size: {}".format(self.EFP_size))
            print("=======================================================\n\n")
            
        # prepare arrays for event & jet features, EFPs and jet constituents
        
        self.output_arrays = {
            OutputTypes.EventFeatures: np.empty((self.n_events, len(Event.get_features_names()))),
            OutputTypes.JetFeatures: np.empty((self.n_events, self.max_n_jets, len(Jet.get_feature_names()))),
            OutputTypes.JetConstituents: np.empty((self.n_events, self.max_n_jets, self.max_n_constituents, len(Jet.get_constituent_feature_names()))),
            OutputTypes.EPFs: np.empty((self.n_events, self.max_n_jets, self.EFP_size))
        }
        
        self.output_names = {
            OutputTypes.EventFeatures: "event_features",
            OutputTypes.JetFeatures: "jet_features",
            OutputTypes.JetConstituents: "jet_constituents",
            OutputTypes.EPFs: "jet_eflow_variables"
        }

        self.output_labels = {
            OutputTypes.EventFeatures: Event.get_features_names(),
            OutputTypes.JetFeatures: Jet.get_feature_names(),
            OutputTypes.JetConstituents: Jet.get_constituent_feature_names(),
            OutputTypes.EPFs: [str(i) for i in range(self.EFP_size)]
        }

        self.save_outputs = {
            OutputTypes.EventFeatures: True,
            OutputTypes.JetFeatures: True,
            OutputTypes.JetConstituents: False if max_n_constituents < 0 else True,
            OutputTypes.EPFs: False if efp_degree < 0 else True
        }
        
        
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
                    self.input_types[path] = InputTypes.Delphes
                    print("Adding Delphes tree")
                elif key.startswith("Events"):
                    self.trees[path] = file[key]
                
                    if file[key]["JetPFCands_eta"] is not None:
                        self.input_types[path] = InputTypes.PFnanoAOD
                    else:
                        self.input_types[path] = InputTypes.nanoAOD
                
                    print("Adding nanoAOD tree: ", key)
                else:
                    print("Unknown tree type: ", key, ". Skipping...")

    def convert(self):
        """
        Reads all selected events from input trees and stores requested features in arrays prepared in the constructor.
        """
        
        total_count = 0
        
        for file_name, tree in self.trees.items():
    
            input_type = self.input_types[file_name]
            data_processor = DataProcessor(tree, input_type)
    
            print("\n\n=======================================================")
            print("Loading events from file: ", file_name)
            print("Input type was recognised to be: ", input_type)

            for iEvent in self.selections[file_name]:
                print("\n\n------------------------------")
                print("Event: ", iEvent)
                
                # load event
                event = Event(data_processor, iEvent, self.jet_delta_r)
                event.print()
                
                # check event properties
                if event.nJets < 2:
                    print("WARNING -- event has less than 2 jets! Skipping...")
                    print("------------------------------\n\n")
                    continue

                if event.has_jets_with_no_constituents(self.max_n_jets):
                    print("WARNING -- one of the jets that should be stored has no constituents. Skipping...")
                    continue
                    
                if not event.are_jets_ordered_by_pt():
                    print("WARNING -- jets in the event are not ordered by pt! Skipping...")
                    continue
                
                # fill feature arrays
                self.output_arrays[OutputTypes.EventFeatures][total_count, :] = np.asarray(event.get_features())

                for iJet, jet in enumerate(event.jets):
                    if iJet == self.max_n_jets:
                        break

                    self.output_arrays[OutputTypes.JetFeatures][total_count, iJet, :] = event.jets[iJet].get_features()

                    if self.save_outputs[OutputTypes.JetConstituents]:
                        self.output_arrays[OutputTypes.JetConstituents][total_count, iJet, :] = jet.get_constituents(self.max_n_constituents)

                    if self.save_outputs[OutputTypes.EPFs]:
                        self.output_arrays[OutputTypes.EPFs][total_count, iJet, :] = jet.get_EFPs(self.efpset)
                        
                total_count += 1
                print("------------------------------\n\n")

            print("\n\n=======================================================")

        # remove redundant rows for events that didn't meet some criteria
        for output_type in OutputTypes:
            for i in range(0, self.n_events - total_count):
                self.output_arrays[output_type] = np.delete(self.output_arrays[output_type], -1, axis=0)
            
    def add_section_to_h5_file(self, file, output_type):
        """
        Adds section with proper name, labels and data to the h5 file, given the output type requested (could be
        event features, jet features, jet constituents etc.)
        """
        if not self.save_outputs[output_type]:
            return
        
        features = file.create_group(self.output_names[output_type])
        features.create_dataset('data', data=self.output_arrays[output_type])
        features.create_dataset('labels', data=self.output_labels[output_type])

    def save(self, output_file_name):
        """
        Creates output h5 file, populates it with data stored in features array and saves it to the disk.
        """
        
        # make sure that the output directory exists and that the file name ends with h5
        path_directory = os.path.dirname(output_file_name)
    
        if not os.path.exists(path_directory) and path_directory is not None and path_directory != '':
            os.mkdir(path_directory)
        
        if not output_file_name.endswith(".h5"):
            output_file_name += ".h5"

        print("\n\n=======================================================")
        print("Saving h5 data to file: ", output_file_name)

        # create output file
        file = h5py.File(output_file_name, "w")

        # add feature arrays to the output file
        for output_type in OutputTypes:
            self.add_section_to_h5_file(file, output_type)
        
        # save the file
        file.close()
        print("Successfully saved!")
        print("=======================================================\n\n")
        