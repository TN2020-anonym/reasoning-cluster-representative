import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def map_clusters(m, n, mapped, factors):
    # Map accidents to clusters
    accidents = mapped[:, -1]
    no_accident_types = np.unique(accidents)
    no_accident_types = no_accident_types.shape[0]
    to_return = np.zeros((m, n, no_accident_types))
    
    for i in range(mapped.shape[0]):
        x = mapped[i, 0]
        y = mapped[i, 1]
        accident = mapped[i, 2]
        to_return[x, y, accident] += 1

    # Map factors to clusters
    no_factors = factors.shape[1]
    factors = np.reshape(factors, (m, n, no_factors))
    to_return = np.concatenate((to_return, factors), axis=2)
    
    return to_return

def plot_distribution(cluster_map, no_accident_types, no_factors, accident_names, factor_names, output):
    def _plot_piechart(accidents_dist, accident_names, display_threshold, plt, ax):
        #plt.figure()
        patches, _, _ = ax.pie(accidents_dist,
                autopct= lambda pct: ('%d' % pct) if pct > display_threshold else '',
                startangle=90,
                shadow=True)
        ax.legend(patches, accident_names, loc='lower right')
        ax.axis('equal')

    # Calculate distributions of accidents
    accidents = cluster_map[:, :, :no_accident_types]
    sum_accidents = np.sum(accidents, axis=2)
    accidents_dists = accidents / sum_accidents[:, :, None]
    accidents_dists *= 100
    accidents_dists = np.nan_to_num(accidents_dists)

    # Calculate impacts of factors
    factors = cluster_map[:, :, no_accident_types:]

    # Calculate minimum and maximum values of factors based on rows as well as columns
    min_factors = np.min(factors, axis=0)
    min_factors = np.min(min_factors, axis=0)
    max_factors = np.max(factors, axis=0)
    max_factors = np.max(max_factors, axis=0)
    
    # Plot data
    m, n = cluster_map.shape[0], cluster_map.shape[1]
    for row in range(m):
        for col in range(n):
            accidents_dist = accidents_dists[row, col]
            print(row, col, list(accidents_dist))
            
            # Ignore the position where there is no accident assigned
            if int(np.sum(accidents_dist)) == 0:
                continue
            accidents_dist = list(accidents_dist)            

            plt.figure(figsize=(50,50))

            # Plot bar chart
            for factor_id in range(len(factor_names)):
                ax = plt.subplot2grid((10, 2), (factor_id*2, 0), rowspan=1, colspan=1)
                factor_name = factor_names[factor_id]
                factor_impact = cluster_map[row, col, no_accident_types + factor_id]
                ax.barh([factor_name], [factor_impact])
                ax.set_xlim(left=min_factors[factor_id], right=max_factors[factor_id])            
            
            # Plot pie chart
            ax = plt.subplot2grid((10,2), (0,1), rowspan=10, colspan=1)            
            _plot_piechart(accidents_dist, accident_names, 10, plt, ax)            

            # Display title, etc.
            figure_name = 'acc_dist_{0}_{1}.png'.format(str(row), str(col))
            title = 'Accident Distribution at row = {0}, col = {1}'.format(row, col)

            plt.suptitle(title)
            plt.savefig(output + figure_name)
            plt.show()
            
def plot_heatmap(cluster_map, start, factor_names, output):
    for col in range(len(factor_names)):
        plt.figure(col + 1)
        sns.heatmap(cluster_map[:, :, start + col], linewidth=.5)
        plt.title(factor_names[col])
        plt.savefig(output + factor_names[col] + '.png')

# Data location
wd = os.path.dirname(os.path.abspath(__file__)) + '/'
output_folder = wd + '../output/'
file_factors = 'cluster_grid.csv'
file_mapped = 'mapped.csv'
file_accidents = 'accidents.csv'

# Read data
accident_names = pd.read_csv(wd + file_accidents)

factors = pd.read_csv(wd + file_factors)
factor_names = factors.columns.values
factor_names = list(factor_names)[1:]

mapped = pd.read_csv(wd + file_mapped)

# Extract data
accident_names = list(accident_names.values[:, -1])
factors = factors.values[:, 1:]
mapped = mapped.values[:, 1:]

# Data characteristics
no_accident_types = np.unique(mapped[:, -1])
no_accident_types = no_accident_types.shape[0]

no_factors = factors.shape[1] - 1

# SOM configuration
m = 20
n = 20

# Map data points to clusters
cluster_map = map_clusters(m, n, mapped, factors)

# Plots for data analysis
plot_distribution(cluster_map, no_accident_types, no_factors, accident_names, factor_names, output_folder + 'dist/')
plot_heatmap(cluster_map, no_accident_types, factor_names, output_folder + 'heatmap/')


print('----')