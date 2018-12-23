## Importing the libraries
import numpy as np
import pandas as pd

class DataReader():
    def __init__(self, data_files, selected_cols):
        self.data_files = data_files
        self.selected_cols = selected_cols
        self.data = np.zeros((1, len(selected_cols)))

    def __read_file(self, file_idx):
        data_file = self.data_files[file_idx]
        dataset = pd.read_csv(data_file, delimiter='\t')
        dataset = dataset.fillna(0)
        data_read = np.hstack([dataset.values[:, col].reshape((-1, 1)) for col in self.selected_cols])
        self.data = np.vstack((self.data, data_read))

    def read(self, file_idx = None):
        if file_idx is None:
            for id in range(len(self.data_files)):
                self.__read_file(id)
        else:
            self.__read_file(file_idx)
        
        self.data = self.data[1:, :]
