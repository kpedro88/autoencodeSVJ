import os
import sys
from argparse import ArgumentParser, ArgumentError
from glob import glob
import re

BASE_COMMAND = 'env -i HOME=$HOME bash -i -c "<CMD>"'
LOG_PREFIX = "Driver :: "
ERROR_PREFIX = LOG_PREFIX + "ERROR: "

### helper functions 

def log(s):
    logbase(s, LOG_PREFIX)

def error(s):
    logbase(s, ERROR_PREFIX)

def logbase(s, prefix):
    if not isinstance(s, str):
        s = str(s)
    for line in s.split('\n'):
        print(prefix + str(line))

def _range_input(s):
    try:
        return tuple(map(int, s.strip().strip(')').strip('(').split(',')))
    except:
        raise ArgumentError("-r input not in format: int,int")

def _smartpath(s):
    if s.startswith('~'):
        return s
    return os.path.abspath(s)

def _check_for_default_file(pathoptions, name, ftype, suffix="txt"):
    
    if isinstance(pathoptions, str):
        pathoptions = [pathoptions]
    
    for path in pathoptions:
        files = os.listdir(path)
        match = []

        for f in files:
            if re.match(r"{0}_[0-9]+_{1}.{2}".format(name, ftype, suffix), f):
                match.append(os.path.join(path, f))

        if len(match) > 0:
            return match

    error("files wih pattern '{0}' does not exist in any of: \n\t{1}".format(name, "\n\t".join(pathoptions)))
    error("quiting")
    raise AttributeError("No files found")

def condor_submit(
    cmd,
    outputdir,
    name,
    setup_cmd = None,
):
    log("writing queue commands")
    run_script = os.path.join(outputdir, "{0}_submit.sh".format(name))
    run_submit = os.path.join(outputdir, "{0}_condor.submit".format(name))
    with open(run_script, 'w+') as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'RUNNING NOW'\n")
        f.write("\n")
        # f.write(cmd + "\n")
        for line in cmd.split("; "):
            f.write("{0}\n".format(line))
    with open(run_submit, 'w+') as f:
        f.write("executable = {0}\n\n".format(run_script))
        f.write("universe = vanilla\n")
        f.write("getenv = True\n")
        f.write("log = {0}\n".format(os.path.join(outputdir, "{0}.log".format(name))))
        f.write("output = {0}\n".format(os.path.join(outputdir, "{0}.out".format(name))))
        f.write("error = {0}\n".format(os.path.join(outputdir, "{0}.err".format(name))))
        f.write("should_transfer_files = YES\n")
        f.write("when_to_transfer_output = ON_EXIT_OR_EVICT\n")
        f.write("transfer_input_files = {0}\n".format(run_script))
        # f.write("transfer_output_files = Data\n")
        f.write("request_cpus = 4\n")
        # f.write("request_disk = 20MB\n")
        # f.write("request_memory = 5MB\n\n")
        f.write("+JobFlavour = \"workday\"\n")
        f.write("queue\n")
    os.system("chmod +rwx {0}".format(run_script))
    condor_cmd = "condor_submit {0}".format(run_submit)
    if setup_cmd is not None:
        condor_cmd = "{0}; ".format(setup_cmd) + condor_cmd
    os.system(condor_cmd)

def local_submit(
    master_command,
):
    os.system(master_command)

def split_to_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def get_data_dict(list_of_selections):
    ret = {}
    for sel in list_of_selections:
        with open(sel, 'r') as f:
            data = [x.strip('\n') for x in f.readlines()]
        for elt in data:
            key, raw = elt.split(': ')
            ret[key] = list(map(int, raw.split()))
    return ret

    # sys.exit(0)

    #     events_parsed = 0
    #     trees_parsed = 0 

    #     with open(spath_path_to_write, "w+") as spath_to_write:
    #         for j,(filespec,spath) in enumerate(zip(filespecs_sub, spaths_sub)):
    #             alldata = ''
    #             with open(spath) as spath_to_read:
    #                 read_events = np.asarray(map(lambda x: map(long, x.split(',')), spath_to_read.read().strip().split()))
    #                                         # spath_to_write.write(read_events)
    #                 # log(" - adding {0} rootfiles to comb. spec, from spec {1}/{2}".format(len(read_lines), j, len(filespecs_sub)))
    #                 # spec_to_write.writelines(read_lines)
    #                 # written += len(read_lines)
                
    #     log("  wrote {0} events to combined selection file, from {1} selection files".format(written, len(spaths_sub)))

    # for i, (filespec, spath) in enumerate(zip(filespecs, spaths)):
    #     log("------------------------------------------")
    #     log("  PERFORMING CONVERSION ON SAMPLE {0}/{1}".format(i, len(filespecs)))
    #     log("------------------------------------------")

    #     sname = name + "_" + str(i)
    #     setup_command = "source " + os.path.abspath("conversion/setup.sh")
    #     python_command = "python " + os.path.abspath("conversion/h5converter.py")
    #     python_command += " " + " ".join(map(str, [outputdir, filespec, spath, sname, DR, n_constituents, range[0], range[1], save_constituents]))

    #     master_command = BASE_COMMAND.replace("<CMD>", "; ".join([setup_command, python_command]))

    #     if dryrun:
    #         log("DRYRUN: command is:")
    #         log(master_command)

    #     elif batch is not None:
    #         if batch == "condor":
    #             condor_setup = ""
    #             condor_submit(condor_setup + master_command, outputdir, name, setup_command)
    #         else:
    #             raise ArgumentError("unrecognized batch platform '{0}'".format(batch))
        
    #     else:
    #         os.system(master_command)

    # sys.exit(0)

### parser setup

def setup_parser():
    parser = ArgumentParser()

    # add subparsers
    subparsers = parser.add_subparsers(dest='COMMAND')

    select = subparsers.add_parser("select", help="base selection of events")
    convert = subparsers.add_parser("convert", help="convert raw root data to h5 files based on selection output")
    train = subparsers.add_parser("train", help="training command for module")

    # shared args
    for subparser in [select, convert, train]:
        subparser.add_argument('-o', '--output', dest="outputdir", action="store", type=_smartpath, help="output dir path", required=True)
        subparser.add_argument('-n', '--name', dest='name', action='store', default='data', help='sample save name')
        subparser.add_argument('-j', '--batch-job', dest='batch', action='store', default=None, help='attempt to run as a batch job on the indicated service')
        subparser.add_argument('-z', '--dry',  dest='dryrun', action='store_true', default=False, help='don\'t run analysis code')
    
    # selection args
    select.add_argument('-s', '--split-trees',  dest='split', action='store', type=int, default=-1, help='split trees into chunks of N')
    select.add_argument('-i', '--input', dest="inputdir", action="store", type=_smartpath, help="input dir path", required=True)
    select.add_argument('-f', '--filter', dest='filter', action='store', default='*', help='glob-style filter for root files in inputfile')
    select.add_argument('-r', '--range', dest='range', action='store', default=(-1,-1), type=_range_input, help='subset of tree values to parse')
    select.add_argument('-d', '--no-debug', dest='debug', action='store_false', default=True, help='disable debug output')
    select.add_argument('-t', '--no-timing', dest='timing', action='store_false', default=True, help='disable timing output')
    select.add_argument('-c', '--no-save-cuts', dest='cuts', action='store_false', default=True, help='disable saving cut values')
    select.add_argument('-b', '--build', dest='build', action='store_true', default=False, help='rebuild cpp files before running')
    select.add_argument('-g', '--gdb', dest='gdb', action='store_true', default=False, help='run with gdb debugger :-)')
    # select.add_argument('-m', '--merge', dest='merge', action='store', type=int, default=-1, help='merge output data by tree groups of N')

    # conversion args
    convert.add_argument('-i', '--input', dest="inputdir", action="store", type=_smartpath, help="input dir path", required=False, default=None)    
    convert.add_argument('-d', '--dr', dest='DR', action='store', type=float, default=0.8, help='dr parameter for jet finding')
    convert.add_argument('-c', '--constituents', dest='NC', action='store', type=int, default=-1, help='number of jet constituents to save')
    convert.add_argument('-r', '--range', dest='range', action='store', type=_range_input, default=(-1,-1), help='range of data to parse')
    convert.add_argument('-s', '--split-samples',  dest='split', action='store', type=int, default=-1, help='split samplelist processing into chunks of N')
    convert.add_argument('-b', '--basis-degree',  dest='basis_n', action='store', type=int, default=-1, help='degree of energy flow basis; -1 is default (don\'t save')
    
    # training args

    return parser

# MAIN functions:

def select_main(inputdir, outputdir, name, batch, filter, range, debug, timing, cuts, build, dryrun, gdb, split):
    log("running command 'select'")
    
    ffilter = str(filter)
    rng = range

    if not ffilter.endswith(".root"):
        ffilter += ".root"

    # get list of samples, write to text file
    criteria = os.path.join(inputdir, ffilter)
    all_samplenames = glob(criteria)

    if len(all_samplenames) == 0:
        error("No samples found matching glob crieria '{0}'".format(criteria))
        sys.exit(1)

    if split < 0:
        split = len(all_samplenames)
    
    split_samplenames = list(split_to_chunks(all_samplenames, split))


    log("running {0} jobs with {1} rootfiles each".format(len(split_samplenames), split))
    log("splits: {0}".format(list(map(len, split_samplenames))))

    if not os.path.exists(outputdir):
        log("making ouput directory '{0}'".format(outputdir))
        os.makedirs(outputdir)

    for i,samplenames in enumerate(split_samplenames):
        log("------------------------------------------")
        log("  PERFORMING SELECTION ON SAMPLE {0}/{1}".format(i + 1, len(split_samplenames)))
        log("------------------------------------------")

        name_sample = name + ('_' + str(i))
        samplefile = os.path.join(outputdir, "{0}_filelist.txt".format(name_sample))
        with open(samplefile, "w+") as sf:
            log("writing samplefile to file '{0}'".format(samplefile))
            for samplename in samplenames:
                sf.write(samplename + '\n')

        path = os.path.abspath(os.path.dirname(__file__))
        setup_command = "source {0}".format(os.path.join(path, "selection/setup.sh"))

        if build:
            setup_command += "; cd {0}; cd ../..; scram b -j 10; cd {1}".format(path, path)
            
        run_command = 'cd {0}; ../../bin/sl*/SVJselection '.format(path) + ' '.join([samplefile, name_sample, outputdir] + list([str(int(x)) for x in [debug, timing, cuts, rng[0], rng[1]]]))
        master_command = setup_command + "; " + run_command

        if dryrun:
            log("DRYRUN: command is:")
            log(master_command)
        
        elif batch is not None:
            if batch == "condor":
                condor_setup = "; ".join([
                    "source /cvmfs/cms.cern.ch/cmsset_default.sh",
                    "export SCRAM_ARCH=slc7_amd64_gcc530",
                    "eval `scramv1 runtime -sh`; ",
                ])
                condor_submit(condor_setup + master_command, outputdir, name_sample, setup_command)
            else:
                raise ArgumentError("unrecognized batch platform '{0}'".format(batch))

        else:
            
            master_command = BASE_COMMAND.replace("<CMD>", master_command)
            local_submit(master_command)

    sys.exit(0)

def convert_main(inputdir, outputdir, name, batch, range, DR, NC, dryrun, split, basis_n):
    log("running command 'convert'")

    if inputdir is None:
        inputdir = outputdir

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    # filespecs = _check_for_default_file([inputdir], name, "filelist")
    spaths = _check_for_default_file([inputdir], name, "selection")
    # errors = _check_for_default_file([inputdir], name, "")
    
    # assert len(filespecs) == len(spaths), "must have equal amounts of filespecs and paths"

    save_constituents = 1
    n_constituents = NC
    
    if n_constituents < 0:
        n_constituents = 100
        save_constituents = 0

    all_data = get_data_dict(spaths)

    if split < 0:
        split = int(len(all_data)/len(spaths))

    split_keys = list(split_to_chunks(list(all_data.keys()), split))

    for i,keys in enumerate(split_keys):

        log("------------------------------------------")
        log("  PERFORMING CONVERSION ON SAMPLE {0}/{1}".format(i + 1, len(split_keys)))
        log("------------------------------------------")

        sname = "{0}_{1}".format(name, i)
        process_name = "{0}_combined.txt".format(sname)
        process_path = os.path.join(outputdir, process_name)
        with open(process_path, 'w+') as f:
            for k in keys:
                s = k + ': '
                s += ' '.join(map(str, all_data[k]))
                s += '\n'
                f.write(s)

        setup_command = "source " + os.path.abspath("conversion/setup.sh")
        python_command = "python " + os.path.abspath("conversion/h5converter.py")
        python_command += " " + " ".join(map(str, [outputdir, process_path, sname, DR, n_constituents, range[0], range[1], save_constituents, basis_n]))

        master_command = BASE_COMMAND.replace("<CMD>", "; ".join([setup_command, python_command]))

        if dryrun:
            log("DRYRUN: command is:")
            log(master_command)

        elif batch is not None:
            if batch == "condor":
                condor_setup = ""
                condor_submit(condor_setup + master_command, outputdir, sname, setup_command)
            else:
                raise ArgumentError("unrecognized batch platform '{0}'".format(batch))
        
        else:
            os.system(master_command)

    sys.exit(0)
     
def train_main(outputdir, name, batch, filter):
    log("running command 'train'")
    sys.exit(0)

if __name__=="__main__":
    log("setting up driver")
    parser = setup_parser()
    argv = sys.argv[1:]

    if len(argv) > 0:
        args = parser.parse_args(argv)
    else:
        parser.print_help()
        sys.exit(0)

    args_dict = vars(args)
    cmd = args_dict["COMMAND"]
    del args_dict["COMMAND"]
    # args_dict = { var: vars(args)[var] for var in vars(args) if "COMMAND" not in var }

    if cmd == "select":
        select_main(**args_dict)
    if cmd == "convert":
        convert_main(**args_dict)
    if cmd == "train":
        train_main(**args_dict) 
    