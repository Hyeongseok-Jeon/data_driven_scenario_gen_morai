#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 04:18:25 2020

@author: jhs
"""
def data_list_load():
    file_list = os.listdir('data/landmark')
    file_list_int = np.zeros(len(file_list), dtype = int)
    for i in range(len(file_list)):
        file_list_int[i] = int(file_list[i][0:4])
        
    return file_list_int

def data_load(data_id):
    background = mpimg.imread('data/background/'+str(data_id)+'.jpg')
        
    return background

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print('Starting KAIST dataset viewer and replayer')
print('Data list loading ...\n')

file_list_int = data_list_load()
print('------------------------------------------------------------')
for i in range(len(file_list_int)):
    print('File_id : ' + str(file_list_int[i]), '  File_index : '+str(i))
print('------------------------------------------------------------')
print('\n')
selected_file_index = input('Select data file index from above :')

while True:
    try:
        selected_file_index = int(selected_file_index)
        if selected_file_index < len(file_list_int)-1:
            break
        else:
            print('wrong data file index')
            selected_file_index = input('Select data file index from above :')
    except:
        print('wrong data file index')
        selected_file_index = input('Select data file index from above :')
selected_scenario_id = file_list_int[selected_file_index]
print('scenario '+str(selected_scenario_id) + ' is selected')

print('\n')
print('Data loading ....')

background = data_load(selected_scenario_id)
imgplot = plt.imshow(background)
plt.show()


