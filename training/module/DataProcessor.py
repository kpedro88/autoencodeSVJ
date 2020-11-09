from module.DataTable import DataTable

from sklearn.model_selection import train_test_split
import numpy as np


class DataProcessor():
    
    def __init__(self, validation_fraction=None, test_fraction=None, seed=None):
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.seed = seed

    def split_to_train_validate_test(self, data_table, n_skip=2, train_idx=None, test_idx=None):
        
        if train_idx is None or test_idx is None:
            train_idx, test_idx = train_test_split(data_table.df.index,
                                                   test_size=self.test_fraction,
                                                   random_state=self.seed)

        train = np.asarray([train_idx]).T.flatten()
        test = np.asarray([test_idx]).T.flatten()
        
        train_and_validation_data = DataTable(data_table.df.loc[train])
        test_data = DataTable(data_table.df.loc[test], name="test")

        train, test = train_test_split(train_and_validation_data,
                                       test_size=self.validation_fraction,
                                       random_state=self.seed)
        
        train_data =  DataTable(train, name="train")
        validation_data = DataTable(test, name="validation")
        
        return train_data, validation_data, test_data, train_idx, test_idx

    def normalize(self, data_table, normalization_type, data_ranges=None, inverse=False, norm_args=None):
        
        if normalization_type == "Custom":
            if data_ranges is None:
                print("Custom normalization selected, but no data ranges were provided!")
                exit(0)
            
            if not inverse:
                return data_table.normalize_in_range(rng=data_ranges)
            else:
                return data_table.inverse_normalize_in_range(rng=data_ranges)
        
        elif normalization_type in ["RobustScaler", "MinMaxScaler", "StandardScaler", "MaxAbsScaler"]:
            return data_table.normalize(norm_type=normalization_type,
                                        scaler_args=norm_args,
                                        inverse=inverse)
        
        else:
            print("ERROR -- Normalization not implemented: ", normalization_type)
            exit(0)
