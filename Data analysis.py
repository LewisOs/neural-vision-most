#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:34:08 2018

@author: Lewis
"""

# importing the libraries

import numpy as np
import pandas as pd

# importing the data sets

RAM_data = pd.read_csv('ram_cnn_g8_randomloc_fulltrain_test_predict.csv')
RAM_data['network_type'] = 'RAM'

vgg7_data = pd.read_csv('vgg7_results.csv')
vgg7_data['network_type'] = 'vgg7'
vgg9_data = pd.read_csv('vgg9_results.csv')
vgg9_data['network_type'] = 'vgg9'
vgg11_data = pd.read_csv('vgg11_results.csv')
vgg11_data['network_type'] = 'vgg11'
vgg13_data = pd.read_csv('vgg13_results.csv')
vgg13_data['network_type'] = 'vgg13'

CNN_data = pd.concat([vgg7_data, vgg9_data, vgg11_data, vgg13_data])

# preprocessing the data

def preprocess(dataset):
    dataset['accuracy'] = (dataset['predicted_class'] == dataset['true_label']).astype(int)
    dataset['absolute_dif'] = abs(dataset['num_blue'] - dataset['num_yellow'])
    if (dataset['num_blue'] < dataset['num_yellow']):
        dataset['ratio_as_num'] = (dataset['num_blue'] / dataset['num_yellow'])
    else:
        dataset['ratio_as_num'] = (dataset['num_yellow'] / dataset['num_blue'])
    
    return dataset

preprocess(RAM_data)
preprocess(CNN_data)

# DESCRIPTIVES

def descriptives(dataset):
    general = dataset.groupby(['network_type']).describe()
    by_trial =  dataset.groupby(['network_type','trial_type']).describe()
    by_ratio = dataset.groupby(['network_type','ratio']).describe()
    by_absolute = dataset.groupby(['network_type','absolute_dif']).describe()
    by_trial_x_ratio = dataset.groupby(['network_type','trial_type','ratio']).describe()
    
    return general['accuracy'], by_trial['accuracy'], by_ratio['accuracy'], by_absolute['accuracy'], by_trial_x_ratio['accuracy']

descriptives(RAM_data)
descriptives(CNN_data)

# INFERENTIALS

""" to do...
    add function for a mixed design ANOVA with:
    DV - accuracy
    IV (within subjects) - trial_type, ratio, absolute_dif
    IV (between subjects) - network type """