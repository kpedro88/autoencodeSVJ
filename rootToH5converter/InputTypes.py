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
                "Jet_eta": "Jet/Jet.Eta",
                "Jet_phi": "Jet/Jet.Phi",
                "Jet_pt": "Jet/Jet.PT",
                "Jet_mass": "Jet/Jet.Mass",
                "Jet_nCharged": "Jet/Jet.NCharged",
                "Jet_nNeutral": "Jet/Jet.NNeutrals",
                "Jet_flavor": "Jet/Jet.Flavor"
            },
            InputTypes.nanoAOD: {
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
            },
            InputTypes.PFnanoAOD: {
                "Jet_eta": "Jet_eta",
                "Jet_phi": "Jet_phi",
                "Jet_pt": "Jet_pt",
                "Jet_mass": "Jet_mass",
                "Jet_chHEF": "Jet_chHEF",
                "Jet_neHEF": "Jet_neHEF",
            }
        }

    def get_value_from_tree(self, value, iEvent, iEntry):
        if value not in self.variables[self.input_type]:
            return None
        
        value = self.variables[self.input_type][value]
        
        if value in self.tree.keys():
            return self.tree[value].array()[iEvent][iEntry]
        
        return None
