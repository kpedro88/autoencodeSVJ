# import autoencodeSVJ.models as models
# import autoencodeSVJ.utils as utils
# import autoencodeSVJ.trainer as trainer

import os

def _get_args():
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

def make_parser():
    import argparse

    def figsize(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) == 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,<int>".format(s))

    def node_arch(s):
        try:
            slist = list(map(int, s.split(",")))
            assert len(slist) > 2
            return tuple(slist)
        except:
            raise argparse.ArgumentTypeError("input '{}' is not of the required format <int>,...,<int>,...,<int>".format(s))

    import sys

    p = argparse.ArgumentParser("python train.py")

    p.add_argument("NAME")
    subp = p.add_subparsers("COMMAND")

    make = subp.add_parser("make")
    train = subp.add_parser("train")
    plot = subp.add_parser("plot")
    

    make.add_argument("-a", "--arch", type=node_arch, required=True, help="node architecture for autoencoder")
    make.add_argument("")

    train.add_argument("-e", "--epochs", type=int)

    plot_types = plot.add_subparsers()

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("-n", "--neck-nodes", dest="bn", required=True, type=int)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=10)
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("-r", "--learning-rate", dest="lr", type=float, default=0.01)
    parser.add_argument("-t", "--norm-type", dest="ntype", type=str, default="RobustScaler")
    parser.add_argument("-l", "--loss", dest="loss", type=str, default="mse")
    parser.add_argument("-p", "--path", dest="path", type=str, default="CWD")


    args = parser.parse_args(sys.argv[1:])

    name = args.name
    bn = args.bn
    epochs = args.epochs
    batch = args.batch_size
    lr = args.lr
    ntype = args.ntype
    loss = args.loss
    path = args.path

    if path == "CWD":
        path = os.path

    import autoencodeSVJ.utils as utils
    import autoencodeSVJ.models as models
    import autoencodeSVJ.trainer as trainer
    import glob
    import os
    import numpy as np

    data,jet_tables = utils.get_training_data_jets("../../data/dijet_tight/*data.h5")
    train, test = data.train_test_split(0.3)

    train_norm, test_norm = data.norm(train, norm_type=ntype), data.norm(test, norm_type=ntype)

    ae_skeleton = models.base_autoencoder()
    ae_skeleton.add(7)
    ae_skeleton.add(30)
    ae_skeleton.add(bn, 'relu')
    ae_skeleton.add(30)
    ae_skeleton.add(7, "linear")
    autoencoder = ae_skeleton.build()
    encoder, decoder = autoencoder.layers[1:]

    name = name + "_" + str(bn)
    path = os.path.join(path, name)
    if os.path.exists(path):
        raise AttributeError("Data at '' already exists!".format(path))

    instance = trainer.trainer(path)

    from keras import backend as K

    def r_square(y_true, y_pred):
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return ( 1 - SS_res/(SS_tot + K.epsilon()))

    # train, test = data.train_test_split(0.25)
    # ntype="RobustScaler"
    # train_norm, test_norm = data.norm(train, norm_type=ntype), data.norm(test, norm_type=ntype)

    autoencoder = instance.train(
        x_train=train_norm.data,
        x_test=test_norm.data,
        y_train=train_norm.data,
        y_test=test_norm.data,
        optimizer="adam",
        loss=loss,
        epochs=epochs,
        model=autoencoder,
        metrics=[r_square, "mae", "mse"],
        force=False,
        batch_size=batch,
        use_callbacks=True,
        learning_rate=lr,
    )

    instance.plot_metrics(fnmatch_criteria="*loss*", yscale="linear")
    # instance.plot_metrics(fnmatch_criteria="*absolute*", yscale="linear")
    # instance.plot_metrics(fnmatch_criteria="*r_square*", yscale="linear")

    data_recon_norm = utils.data_table(autoencoder.predict(data.norm(norm_type=ntype).df.values), headers=train_norm.headers)
    data_recon = data.inorm(data_recon_norm, norm_type=ntype)
    data_recon.name = "all jet data (pred)"

    data.plot(
        data_recon,
        normed=0, bins=35, alpha=1.0, figloc="upper right",
        figsize=(25,15), fontsize=22, rng=[(-2,2), (-4,4), (0,2000), (0,210), (0,1), (0,0.8), (0, .10)]
    )

    encoder, decoder = autoencoder.layers[1:]
    train_reps, test_reps = (
        utils.data_table(encoder.predict(train_norm.data), name="train_reps"),
        utils.data_table(encoder.predict(test_norm.data), name="test_reps")
    )
    train_reps.plot([test_reps], cols=5, figsize=(25,5), fontsize=22, normed=1, bins=40)
    # data.head()
    # args = get_args()
    # print args

