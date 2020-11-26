#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 04:18:25 2020

@author: jhs
"""


def data_list_load():
    file_list = os.listdir('../data/landmark')
    file_list_int = np.zeros(len(file_list), dtype=int)
    for i in range(len(file_list)):
        file_list_int[i] = int(file_list[i][0:4])

    return file_list_int


def data_load(data_id):
    background = mpimg.imread('../data/background/' + str(data_id) + '.jpg')
    landmark = np.genfromtxt('../data/landmark/' + str(data_id) +'_landmarks.csv', skip_header=1, delimiter=',',dtype = int)
    landmark[:,1] = landmark[:,1] % 10000
    recordingMeta = np.genfromtxt('../data/recordingMeta/' + str(data_id) + '_recordingMeta.csv', skip_header=1, delimiter = ',')
    recordingMeta[3] = recordingMeta[3] % 10000
    tracks = np.genfromtxt('../data/tracks/' + str(data_id) + '_tracks.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.genfromtxt('../data/tracksMeta/' + str(data_id) + '_trackMeta.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.delete(tracksMeta, -1, -1)
    tracksClass = []
    with open('../data/tracksMeta/' + str(data_id) + '_trackMeta.csv', "r") as tmp_file:
        csvReader = csv.reader(tmp_file)
        header = next(csvReader)
        class_index = header.index("class")
        for row in csvReader:
            class_tmp = row[class_index]
            tracksClass.append(class_tmp)

    return background, landmark, recordingMeta, tracks, tracksMeta, tracksClass

def coordinate_conversion(tracks, landmark, recordingMeta, origin_GT):
    meter_per_pixel = recordingMeta[15]
    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    for i in range(len(tracks)):
        landmark1 = [landmark[i, 2] * meter_per_pixel, -landmark[i, 3] * meter_per_pixel]
        landmark2 = [landmark[i, 4] * meter_per_pixel, -landmark[i, 5] * meter_per_pixel]
        landmark3 = [landmark[i, 6] * meter_per_pixel, -landmark[i, 7] * meter_per_pixel]
        landmark4 = [landmark[i, 8] * meter_per_pixel, -landmark[i, 9] * meter_per_pixel]
        a = np.array([landmark1,landmark2, landmark3, landmark4])

        landmark1_GT = origin_GT[0]
        landmark2_GT = origin_GT[1]
        landmark3_GT = origin_GT[2]
        landmark4_GT = origin_GT[3]
        b = np.array([landmark1_GT, landmark2_GT, landmark3_GT, landmark4_GT])

        conversion = np.linalg.lstsq(a,b,rcond='warn')
        new_tracks[:,4:6] * conversion[0]
        


    return new_tracks

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import lcm

print('Starting KAIST dataset viewer and replayer')
print('Data list loading ...\n')

file_list_int = data_list_load()
print('------------------------------------------------------------')
for i in range(len(file_list_int)):
    print('File_id : ' + str(file_list_int[i]), '  File_index : ' + str(i))
print('------------------------------------------------------------')
print('\n')
selected_file_index = input('Select data file index from above :')

while True:
    try:
        selected_file_index = int(selected_file_index)
        if selected_file_index < len(file_list_int) - 1:
            break
        else:
            print('wrong data file index')
            selected_file_index = input('Select data file index from above :')
    except:
        print('wrong data file index')
        selected_file_index = input('Select data file index from above :')
selected_scenario_id = file_list_int[selected_file_index]
print('scenario ' + str(selected_scenario_id) + ' is selected')

print('\n')
print('Data loading ....')

background, landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(selected_scenario_id)
origin_GT = [[landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 2] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 3] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 4] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 5] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 6] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 7] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 8] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 9] * recordingMeta[15]]]


imgplot = plt.imshow(background)
plt.show()

