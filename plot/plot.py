import sys
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', help="file to edit")
    parser.add_argument('-x', '--x-max', dest='x', help='x axis max', default=-1, type=int)
    parser.add_argument('-y', '--y-max', dest='y', help='y axis max', default=-1, type=int)
    parser.add_argument('-o', '--output', dest='out', help='output path', default='.', type=str)
    args = parser.parse_args(sys.argv[1:])
    out = os.path.abspath(args.out) + '/'
    curpath = os.path.abspath(os.path.dirname(__file__))
    plotfile = os.path.join(curpath, "plot.cpp")
    rf = args.FILE
    if not rf.endswith(".root"):
        rf += ".root"
    if not os.path.exists(rf):
        import glob
        rf = rf.rstrip(".root")
        files = glob.glob(os.path.join(rf, "*_output.root"))
        if len(files) == 1:
            rf = os.path.abspath(files[0])
        else:
            assert False, "FILE '{0}' given does not exist!".format(args.FILE)
    print("working on rootfile", rf)
    
    os.system("""root -b -q '{4}("{0}", "{3}", {1}, {2})'""".format(rf, args.x, args.y, out, plotfile))
    