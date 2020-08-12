# import autoencodeSVJ.utils as utils
# import autoencodeSVJ.skeletons as skeletons
# import autoencodeSVJ.models as models

def get_args():
    import argparse

    def figsize(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) == 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,<int>".format(s))

    import sys
    parser = argparse.ArgumentParser("python plot.py")

    parser.add_argument("-n", "--name", help="name for the training instance", required=True)
    sub = parser.add_subparsers(dest="PLOT_COMMAND")

    reps = sub.add_parser("reps", help="plot reps for the model specified")
    data = sub.add_parser("data", help="plot input/output data for model specified")
    loss = sub.add_parser("loss", help="plot loss statistics for model specified")
    
    reps.add_argument("-d", "--data", help="path to data to parse", required=True)
    data.add_argument("criteria", help="glob criteria for data subsets to plot")
    reps.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    reps.add_argument("-y", "--yscale", default="linear")
    reps.add_argument("-x", "--xscale", default="linear")
    reps.add_argument("-c", "--columns", default=4, type=int)
    # reps.add_argument("")
    
    data.add_argument("-d", "--data", help="path to data to parse", required=True)
    data.add_argument("criteria", help="glob criteria for data subsets to plot")
    data.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    data.add_argument("-y", "--yscale", default="linear")
    data.add_argument("-x", "--xscale", default="linear")
    data.add_argument("-c", "--columns", default=4, type=int)

    loss.add_argument("criteria", help="glob criteria for losses to plot")
    loss.add_argument("-y", "--yscale", default="linear")
    loss.add_argument("-x", "--xscale", default="linear")
    loss.add_argument("-s", "--figsize", default=(7,5), type=figsize)
    loss.add_argument("-c", "--columns", default=4, type=int)

    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    raise NotImplementedError("yEAh nO don't use this yet please")
    args = get_args()
    print(args)