## Importing the libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from utils.DataParser import *
from utils.DataReader import *
from utils.DataScaler import *
from model.SOM import *

def getUniqueAccidentType(accidents):
    accident_type = set()
    for accident in accidents:
        accident_type.add(accident)

    # Remove an event where no accident takes place
    if 0 in accident_type:
        accident_type.remove(0)

    return accident_type

if __name__ == "__main__":
    # Data location
    data_path = os.path.dirname(os.path.abspath(__file__))
    data_path += '/data/'
    data_path += 'prototype/'

    data_files = os.listdir(data_path)
    for i in range(len(data_files)):
        data_files[i] = data_path + data_files[i]

    # Data definition by columns
    #   = cause: datetime, rainfall_max, congestion_max_length, sns_message_id
    #   = result: accident_accident_type
    read_col = [1, 5, 10, 11, 15]
    target_col = 4

    # Read data
    dr = DataReader(data_files, read_col)
    dr.read()
    data = dr.data

    # Parse data
    dp = DataParser()
    ds = DataScaler()
    data = dp.convertDateTime(data, 0, ('-', ' ', ':', '+09'))
    data = ds.extractTime(data, 0)
    data = dp.convertInt(data, 1)
    data = dp.convertInt(data, 2)
    data = dp.convertList(data, 3, '\{\}', ',')
    print(data[500, :])

    # Cleanup data, use only event where an accident takes place
    data = data[data[:, 4] != 0]    

    # Split data    
    X = data[:, :target_col]
    y = data[:, target_col]    
    alias = list(np.unique(y))
    y = dp.convertTextTarget(y, alias)
    print(y)
    # Standardize data
    X = ds.standardize(X)    

    # To train/test
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.33, random_state=42)
    #print(X.shape, y.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = X

    #Train SOM
    m = 20
    n = 20
    n_iter = 2000
    som = SOM(m, n, len(read_col) - 1, n_iter)
    som.train(X_train)
    
    #Get output grid
    cluster_grid = som.get_centroids()

    #Map colours to their closest neurons
    mapped = som.map_vects(X_train)
    
    #Plot
    plt.imshow(cluster_grid)
    plt.title('Accident SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], y[i], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()
    
    print('end')
    '''if __name__ == "__main__":
    #For plotting the images
    from matplotlib import pyplot as plt
    
    #Training inputs for RGBcolors
    colors = np.array(
        [[0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0.5],
        [0.125, 0.529, 1.0],
        [0.33, 0.4, 0.67],
        [0.6, 0.5, 1.0],
        [0., 1., 0.],
        [1., 0., 0.],
        [0., 1., 1.],
        [1., 0., 1.],
        [1., 1., 0.],
        [1., 1., 1.],
        [.33, .33, .33],
        [.5, .5, .5],
        [.66, .66, .66]])
    color_names = \
        ['black', 'blue', 'darkblue', 'skyblue',
        'greyblue', 'lilac', 'green', 'red',
        'cyan', 'violet', 'yellow', 'white',
        'darkgrey', 'mediumgrey', 'lightgrey']
    
    #Train a 20x30 SOM with 400 iterations
    som = SOM(20, 30, 3, 400)
    som.train(colors)
    
    #Get output grid
    image_grid = som.get_centroids()
    
    #Map colours to their closest neurons
    mapped = som.map_vects(colors)
    
    #Plot
    plt.imshow(image_grid)
    plt.title('Color SOM')
    for i, m in enumerate(mapped):
        plt.text(m[1], m[0], color_names[i], ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, lw=0))
    plt.show()'''
    