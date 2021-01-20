import uproot
from Converter import Converter

file = uproot.open("/Users/Jeremi/Documents/Physics/ETH/data/s_channel_delphes/qcd/qcd_sqrtshatTeV_13TeV_PU20_9.root")

converter = Converter(input_paths = ["test_selections.txt"],
                      output_path= "./",
                      output_file_prefix= "qcd",
                      save_constituents=False,
                      energyflow_basis_degree=-1,
                      n_constituent_particles=100
                      )

converter.convert((0,100))
converter.save("test.h5")

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


