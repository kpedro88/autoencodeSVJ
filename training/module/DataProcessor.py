from module.DataTable import DataTable

from sklearn.model_selection import train_test_split
import numpy as np


class DataProcessor():
    
    def __init__(self, validation_fraction, test_fraction, seed):
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.seed = seed

    def split_to_train_validate_test(self, data_table, n_skip=2):
        train_idx, test_idx = train_test_split(data_table.df.index[0::n_skip],
                                               test_size=self.test_fraction,
                                               random_state=self.seed)
        
        train, test = [np.asarray([x + i for i in range(n_skip)]).T.flatten() for x in [train_idx, test_idx]]
        
        train_and_validation_data = DataTable(data_table.df.loc[train])
        test_data = DataTable(data_table.df.loc[test], name="test")

        train, test = train_test_split(train_and_validation_data,
                                       test_size=self.validation_fraction,
                                       random_state=self.seed)
        
        train_data =  DataTable(train, name="train")
        validation_data = DataTable(test, name="validation")
        
        return train_data, validation_data, test_data
        
        


    
    
    
    