from Converter import Converter
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-i", "--input", dest="input_path", default=None,
                    help="path to text file with ROOT files' paths and selected events")

parser.add_argument("-o", "--output", dest="output_path", default="output.h5",
                    help="output file name (default: output.h5)")

parser.add_argument("-c", "--constituents", dest='save_constituents', action='store_true',
                    help="If specified, jte constituents will be stored in the output file.")

parser.add_argument("-m", "--max_constituents", dest="max_constituents", type=int, default=100,
                    help="Maximum number of constituents per jet to be stored (default: 100).")

parser.add_argument("-e", "--efp_basis_degree", dest="efp_basis_degree", type=int, default=-1,
                    help="EFPs degree to be calculated and stored in the output file (default: not stored)")

args = parser.parse_args()

if args.input_path is None:
    parser.print_help()
    exit(0)

print("\n\n=======================================================")
print("Running ROOT to h5 converter with the following options: ")
print("input: ", args.input_path)
print("output: ", args.output_path)
print("save constituents: ", args.save_constituents)
print("max constituents: ", args.max_constituents)
print("EFP basis degree: ", args.efp_basis_degree)
print("=======================================================\n\n")

converter = Converter(input_path = args.input_path,
                      save_constituents=args.save_constituents,
                      energyflow_basis_degree=args.efp_basis_degree,
                      max_n_constituents=args.max_constituents
                      )

converter.convert()
converter.save(args.output_path)


