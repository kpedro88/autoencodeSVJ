import autoencodeSVJ.utils as utils
import autoencodeSVJ.trainer as trainer
import autoencodeSVJ.evaluate as ev

reps = 3
for norm_type in ['StandardScaler', 'MinMaxScaler']:
    for target_dim in [9,10,11]:
        print('running reps with...')
        print((' - norm_type: {}'.format(norm_type)))
        print((' - target_dim: {}'.format(target_dim)))
        for i in range(reps):
            auc = ev.ae_train(
                qcd_path='data/background/base_3/*.h5',
                signal_path='data/signal/base_3/*.h5',
                target_dim=target_dim,
                batch_size=32,
                optimizer='adam',
                norm_args={
                    'norm_type': norm_type
                },
                epochs=100,
                learning_rate=0.0003,
                hlf=1,
                eflow=1,
                verbose=False
            )
            print(("finished training, AUC {:.5f}".format(auc)))
