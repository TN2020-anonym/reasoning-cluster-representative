## Importing the libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

def parse_data(dp, data, col_conf, target_col):
    # Parse data to numeric datatype
    data = dp.convertDateTime(data, col_conf.index('datetime'), ('-', ' ', ':', '+09'))
    data = ds.extractTime(data, col_conf.index('datetime'), isTakeHour=True)
    data = dp.convertInt(data, col_conf.index('meshcode'))
    data = dp.convertInt(data, col_conf.index('rainfall_max'))
    data = dp.convertInt(data, col_conf.index('congestion_max_length'))
    data = dp.convertList(data, col_conf.index('sns_message_id'), '\{\}', ',')

    # Take only events where accidents take place
    data = data[data[:, target_col] != 0]

    return data

def plot_heatmap(data, col_name, wd):
    for id in range(len(col_name) - 1):
        plt.figure(id + 1)
        sns.heatmap(data[:, :, id], linewidth=.5)
        plt.title(col_name[id])
        plt.savefig(wd + col_name[id] + '.png')
        
    #plt.show()

def plot_distance_map(distance, wd):
    plt.figure()
    sns.heatmap(distance, linewidth=.5)
    plt.title('distance')
    plt.savefig(wd + 'distance' + '.png')

def dump_result(path, data, column_names):
    import pandas as pd
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(path)

if __name__ == "__main__":
    # ============================================ #
    # Data definition by columns
    #   = cause: datetime, meshcode, rainfall_max, congestion_max_length, sns_message_id
    #   = result: accident_accident_type
    col_name = ('datetime', 'meshcode', 'rainfall_max', 'congestion_max_length', 'sns_message_id', 'accident_accident_type')
    col_idx  = (1         , 2         , 3             , 4                      , 5               , 6)
    target_col = len(col_name) - 1

    # ============================================ #
    # Data location
    wd = os.path.dirname(os.path.abspath(__file__)) + '/'
    data_path = wd + 'data/'
    output_path = wd + 'output/'

    # ============================================ #
    # Read data
    data_files = os.listdir(data_path)
    for i in range(len(data_files)):
        data_files[i] = data_path + data_files[i]
    
    dr = DataReader(data_files, col_idx)
    ds = DataScaler()
    dp = DataParser()

    print('======== Supplying data ============')
    dr.read()
    
    #dump_result(output_path + 'data_read.csv', dr.getData(), column_names=col_name)

    print('======== Extracting data ============')
    # ============================================ #
    # Split data
    X = dr.data[:, :target_col]
    y = dr.data[:, target_col]    
    alias = list(np.unique(y))
    y = dp.convertTextTarget(y, alias)
    #dump_result(output_path + 'accidents.csv', np.array(alias), ['accident'])
    print('Accident types: ', alias)

    # Standardize data
    X = ds.standardize(X)    

    print('======== Training ============')
    # ============================================ #
    # Train SOM
    m = 20
    n = 20
    n_iter = 15000
    som = SOM(m, n, len(col_name) - 1, n_iter, save=wd + 'checkpoint/', restore=wd + 'checkpoint/')
    som.train(X, checkpoint_len=2)
    
    # ============================================ #
    # Get cluster grid formed by SOM
    cluster_grid = som.get_centroids()
    cluster_grid = np.array(cluster_grid)
    cluster_grid = np.reshape(cluster_grid, (m * n, len(col_name) - 1))
    cluster_grid = ds.inverse_transform(cluster_grid)

    # Dump data to plot on R
    dump_result(output_path + 'cluster_grid.csv', cluster_grid, col_name[:-1]) 
    cluster_grid = np.reshape(cluster_grid, (m, n, len(col_name) - 1))

    # Map colours to their closest neurons
    mapped = som.map_vects(X)
    mapped = np.array(mapped)
    mapped = np.hstack((mapped, y.reshape((-1, 1))))

    # Get distance to neighbours
    distance = som.distance_map()

    
    # ============================================ #
    # Heatmap for datatypes
    plot_heatmap(cluster_grid, col_name, output_path)
    plot_distance_map(distance, output_path)
    dump_result(output_path + 'mapped.csv', mapped, ('X', 'Y', 'target'))
