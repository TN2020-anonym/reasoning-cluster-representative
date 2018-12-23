import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler():
    def __init__(self):
        self.mmScaler = MinMaxScaler()

    def extractTime(self, data, col):
        data_col = data[:, col]
        data_col %= 1000000 # hhmmss
        data[:, col] = data_col

        return data

    def standardize(self, data, isTransformOnly = False):
        if isTransformOnly == True:
            data = self.mmScaler.transform(data)
        else:
            data = self.mmScaler.fit_transform(data)
        return data