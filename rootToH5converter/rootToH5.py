import uproot
from Converter import Converter

file = uproot.open("/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/qcd/qcd_sqrtshatTeV_13TeV_PU20_9.root")

# input_path = "test_selections_delphes.txt"
# output_path = "test_delphes.h5"

# input_path = "test_selections_nanoAOD.txt"
# output_path = "test_nanoAOD.h5"

input_path = "test_selections_PFnanoAOD.txt"
output_path = "test_PFnanoAOD.h5"

# input_path = "data_0_selection.txt"
# output_path = "test_data_0.h5"

converter = Converter(input_paths = [input_path],
                      output_path= "./",
                      output_file_prefix= "qcd",
                      save_constituents=True,
                      energyflow_basis_degree=3,
                      max_n_constituents=100
                      )

converter.convert()
converter.save(output_path)

# print("File keys:", file.keys())
# print("File values:", file.values())
# print("Class names: ", file.classnames())
#
# delphesTree = file["Delphes"]
#
# print("Tree: ", delphesTree)
# print("Tree keys: ", delphesTree.keys())
#
# print("Tree members: ", delphesTree.all_members)
#
# eta = delphesTree["Jet/Jet.Eta"].array()
#
# print("Eta: ", eta)


