import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

training_data = pd.read_csv('isic-2024-challenge/train-metadata.csv', engine='python')
print(training_data.head())



# Define important features
features = [
    'target',
    'age_approx',
    'anatom_site_general',
    'clin_size_long_diam_mm',
    'tbp_tile_type',
    # 'tbp_lv_A',
    # 'tbp_lv_Aext',
    # 'tbp_lv_B',
    # 'tbp_lv_Bext',
    # 'tbp_lv_C',
    # 'tbp_lv_Cext',
    # 'tbp_lv_H',
    # 'tbp_lv_Hext',
    # # 'tbp_lv_L',
    # # 'tbp_lv_Lext',
    # 'tbp_lv_areaMM2',
    # 'tbp_lv_area_perim_ratio',
    # 'tbp_lv_color_std_mean',
    # # 'tbp_lv_deltaA',
    # # 'tbp_lv_deltaB',
    # # 'tbp_lv_deltaL',
    # 'tbp_lv_deltaLBnorm',
    # 'tbp_lv_eccentricity',
    # 'tbp_lv_location',
    # # 'tbp_lv_location_simple',
    # 'tbp_lv_minorAxisMM',
    # 'tbp_lv_norm_border',
    # 'tbp_lv_norm_color',
    # 'tbp_lv_perimeterMM',
    # 'tbp_lv_radial_color_std_max',
    # # 'tbp_lv_stdL',
    # # 'tbp_lv_stdLExt',
    # 'tbp_lv_symm_2axis',
    # 'tbp_lv_symm_2axis_angle',
    # 'tbp_lv_x',
    # 'tbp_lv_y',
    # 'tbp_lv_z'
]
print(training_data.dtypes)


# Plot important features

num_features = len(features)

figure, axes = plt.subplots(num_features, num_features)

# Define the features to plot
plot_data = training_data[features].dropna()

# Create a dictionary mapping lesion malignancy to unique colors
# 0 is benign, 1 is malignant
target_to_color = {0: 'blue', 1: 'red'}

# Plot a scatter for each feature pair.
# Make sure to color the points by their class label.
# Include an x-label and a y-label for each subplot.

print(plot_data[0:20])

for x in range(num_features):
    x_label = features[x]
    for y in range(x+1):
        y_label = features[y]
        for target, color in target_to_color.items():
            data_by_target = plot_data.loc[plot_data['target'] == target]
            axes[x][y].scatter(data_by_target[x_label], data_by_target[y_label], c = color)
        axes[x][y].set_xlabel(x_label)
        axes[x][y].set_ylabel(y_label)

plt.tight_layout()
plt.show()