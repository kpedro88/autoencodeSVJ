from Converter import Converter
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-i", "--input", dest="input_path", default=None, required=True,
                    help="path to text file with ROOT files' paths and selected events")

parser.add_argument("-o", "--output", dest="output_path", default="output.h5",
                    help="output file name (default: output.h5)")

parser.add_argument("-c", "--max_constituents", dest="max_constituents", type=int, default=-1,
                    help="Maximum number of constituents per jet to be stored (default: not stored).")

parser.add_argument("-e", "--EFP_degree", dest="EFP_degree", type=int, default=-1,
                    help="EFPs degree to be calculated and stored in the output file (default: not stored)")

parser.add_argument("-r", "--delta_r", dest="delta_r", type=float, default=0.5,
                    help="Delta R to assign constituents to jets. Only valid for Delphes, for PFnano links between jets and constituents are used")

parser.add_argument("-j", "--store_n_jets", dest="store_n_jets", type=int, default=2,
                    help="Number of jets to be stored (default: 2).")


args = parser.parse_args()


print("\n\n=======================================================")
print("Running ROOT to h5 converter with the following options: ")
print("input: ", args.input_path)
print("output: ", args.output_path)
print("max constituents: ", args.max_constituents)
print("EFP basis degree: ", args.EFP_degree)
print("=======================================================\n\n")

converter = Converter(input_path = args.input_path,
                      store_n_jets= args.store_n_jets,
                      jet_delta_r = args.delta_r,
                      efp_degree=args.EFP_degree,
                      max_n_constituents=args.max_constituents
                      )

converter.convert()
converter.save(args.output_path)


