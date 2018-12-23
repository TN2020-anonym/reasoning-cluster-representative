## Importing the libraries
import numpy as np
import numpy.core.defchararray as np_f
import os
#from DataReader import *

class DataParser():
    def __init__(self):
        pass

    def convertDateTime(self, data, col, removed_str):
        data_col = data[:, col]
        for removed_s in removed_str:
            data_col = [s.replace(removed_s, '') for s in data_col]

        data_col = np.array(data_col)
        data_col = data_col.astype(np.int)

        data[:, col] = data_col

        return data    
    
    def convertList(self, data, col, bound, sep):
        data_col = data[:, col]
        data_col[data_col == 0] = bound
        data_col = [s.count(sep) + 1 for s in data_col]
        data[:, col] = data_col

        return data

    def convertInt(self, data, col):
        data_col = data[:, col]
        data_col = data_col.astype(int)
        data[:, col] = data_col

        return data

    def convertTextTarget(self, data_col, alias):
        for i in range(len(data_col)):
            counter = 0
            for label in alias:            
                counter += 1
                if data_col[i] == label:
                    data_col[i] = counter
                    break
        return data_col
