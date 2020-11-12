import module.utils as utils
import module.Trainer as trainer
import module.AutoEncoderBase as models
import module.SummaryProcessor as summaryProcessor
from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader
import numpy as np
import datetime


class AutoEncoderTrainer:
   
    def __init__(self,
                 qcd_path,
                 training_params,
                 bottleneck_size,
                 intermediate_architecture=(30, 30),
                 test_data_fraction=0.15,
                 validation_data_fraction=0.15,
                 norm_type="",
                 norm_args=None,
                 hlf_to_drop=None,
                 ):
        """
        Creates auto-encoder trainer with random seed, provided training parameters and architecture.
        Loads specified data, splits them into training, validation and test samples according to
        provided arguments. Normalizes the data as specified by norm_percentile.
        High-level features specified in hlf_to_drop will not be used for training.
        """

        if hlf_to_drop is None:
            hlf_to_drop = ['Energy', 'Flavor']
        
        self.seed = np.random.randint(0, 99999999)
        utils.set_random_seed(self.seed)
        
        self.qcd_path = qcd_path
        self.hlf_to_drop = hlf_to_drop
        
        self.training_params = training_params
        self.test_data_fraction = test_data_fraction
        self.validation_data_fraction = validation_data_fraction
        
        data_loader = DataLoader()
        
        # Load QCD samples
        (self.qcd, qcd_jets, qcd_event, qcd_flavor) = data_loader.load_all_data(qcd_path, "qcd background",
                                                                                include_hlf=True, include_eflow=True,
                                                                                hlf_to_drop=hlf_to_drop)

        data_processor = DataProcessor(validation_fraction=self.validation_data_fraction,
                                       test_fraction=self.test_data_fraction,
                                       seed=self.seed)
        
        (train_data,
         validation_data,
         test_data, _, _) = data_processor.split_to_train_validate_test(data_table=self.qcd)
        
        train_data.name = "qcd training data"
        validation_data.name = "qcd validation data"
        
        # Normalize the input
        self.norm_type = norm_type
        self.norm_args = norm_args
        
        self.data_ranges = np.asarray([])

        self.means_train, self.stds_train = None, None
        self.means_validation, self.stds_validation = None, None
        
        if norm_type == "Custom":
            self.data_ranges = utils.percentile_normalization_ranges(train_data, norm_args["norm_percentile"])
        elif norm_type == "CustomStandard":
            self.means_train, self.stds_train = train_data.get_means_and_stds()
            self.means_validation, self.stds_validation = validation_data.get_means_and_stds()
        
        print("Trainer scaler args: ", self.norm_args)
        
        self.train_data_normalized = data_processor.normalize(data_table=train_data,
                                                              normalization_type=self.norm_type,
                                                              norm_args=self.norm_args,
                                                              data_ranges=self.data_ranges,
                                                              means=self.means_train,
                                                              stds=self.stds_train)
        
        self.validation_data_normalized = data_processor.normalize(data_table=validation_data,
                                                                   normalization_type=self.norm_type,
                                                                   norm_args=self.norm_args,
                                                                   data_ranges=self.data_ranges,
                                                                   means=self.means_validation,
                                                                   stds=self.stds_validation)
        
        # Build the model
        self.input_size = len(self.qcd.columns)
        self.intermediate_architecture = intermediate_architecture
        self.bottleneck_size = bottleneck_size
        self.model = self.get_auto_encoder_model()
        
    def run_training(self, training_output_path, summaries_path, verbose=False):
        """
        Runs the training on data loaded and prepared in the constructor, according to training params
        specified in the constructor
        """
        
        self.output_filename = self.get_filename(summaries_path=summaries_path)
        self.training_output_path = training_output_path + self.output_filename
    
        print("\n\nTraining the model")
        print("Filename: ", self.training_output_path)
        print("Number of training samples: ", len(self.train_data_normalized.data))
        print("Number of validation samples: ", len(self.validation_data_normalized.data))
        
        if verbose:
            self.model.summary()
            print("\nTraining params:")
            for arg in self.training_params:
                print((arg, ":", self.training_params[arg]))
        
        self.start_timestamp = datetime.datetime.now()
        
        trainer.Trainer(self.training_output_path, verbose=verbose).train(
            x_train=self.train_data_normalized.data,
            x_test=self.validation_data_normalized.data,
            y_train=self.train_data_normalized.data,
            y_test=self.validation_data_normalized.data,
            model=self.model,
            force=True,
            use_callbacks=True,
            verbose=int(verbose),
            **self.training_params,
            output_path=self.training_output_path
        )
        
        self.end_timestamp = datetime.datetime.now()
        print("Training executed in: ", (self.end_timestamp - self.start_timestamp), " s")
        
    def save_last_training_summary(self, path):
        """
        Dumps summary of the most recent training to a summary file.
        """
        summary_dict = {
            'training_output_path': self.training_output_path,
            'qcd_path': self.qcd_path,
            'hlf': True,
            'hlf_to_drop': tuple(self.hlf_to_drop),
            'eflow': True,
            'eflow_base': self.get_EFP_base(),
            'test_split': self.test_data_fraction,
            'val_split': self.validation_data_fraction,
            'norm_type': self.norm_type,
            'norm_args' : self.norm_args,
            'norm_means_train' : self.means_train,
            'norm_stds_train': self.stds_train,
            'norm_means_validation': self.means_validation,
            'norm_stds_validation': self.stds_validation,
            'range': self.data_ranges.tolist(),
            'target_dim': self.bottleneck_size,
            'input_dim': self.input_size,
            'arch': self.get_architecture_summary(),
            'seed': self.seed,
            'start_time': str(self.start_timestamp),
            'end_time': str(self.end_timestamp),
        }
        summaryProcessor.dump_summary_json(self.training_params, summary_dict, output_path=path)
        
    def get_EFP_base(self):
        """
        Returns EFP base deduced from the structure of QCD samples
        """
        n_EFP_variables = len([x for x in self.qcd.columns if "eflow" in x])
        EFP_base_lookup = {12: 3, 13: 3, 35: 4, 36: 4}
        EFP_base = EFP_base_lookup[n_EFP_variables]
        
        return EFP_base

    def get_filename(self, summaries_path):
        """
        Returns filename for given QCD sample, already with correct next version deduced
        from contents of the provided summaries directory
        """
        
        filename = "hlf_eflow{}_{}_".format(self.get_EFP_base(), self.bottleneck_size)
        last_version = summaryProcessor.get_last_summary_file_version(summaries_path, filename)
        filename += "v{}".format(last_version + 1)
    
        return filename

    def get_auto_encoder_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """
        
        aes = models.AutoEncoderBase()
        aes.add(self.input_size)
        for elt in self.intermediate_architecture:
            aes.add(elt, activation='relu')
        aes.add(self.bottleneck_size, activation='relu')
        for elt in reversed(self.intermediate_architecture):
            aes.add(elt, activation='relu')
        aes.add(self.input_size, activation='linear')
    
        return aes.build()

    def get_architecture_summary(self):
        """
        Returns a tuple with number of nodes in each consecutive layer of the auto-encoder
        """
        
        arch = (self.input_size,) + self.intermediate_architecture
        arch += (self.bottleneck_size,)
        arch += tuple(reversed(self.intermediate_architecture)) + (self.input_size,)
        return arch
        