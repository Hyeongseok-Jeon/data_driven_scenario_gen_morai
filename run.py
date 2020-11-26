#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 04:18:25 2020

@author: jhs
"""

def data_list_load():
    file_list = os.listdir('data/landmark')
    file_list_int = np.zeros(len(file_list), dtype=int)
    for i in range(len(file_list)):
        file_list_int[i] = int(file_list[i][0:4])

    return file_list_int


def data_load(data_id):
    # background = mpimg.imread('../data/background/' + str(data_id) + '.jpg')
    landmark = np.genfromtxt('data/landmark/' + str(data_id) +'_landmarks.csv', skip_header=1, delimiter=',',dtype = int)
    landmark[:,1] = landmark[:,1] % 10000
    recordingMeta = np.genfromtxt('data/recordingMeta/' + str(data_id) + '_recordingMeta.csv', skip_header=1, delimiter = ',')
    recordingMeta[3] = recordingMeta[3] % 10000
    tracks = np.genfromtxt('data/tracks/' + str(data_id) + '_tracks.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.genfromtxt('data/tracksMeta/' + str(data_id) + '_trackMeta.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.delete(tracksMeta, -1, -1)
    tracksClass = []
    with open('data/tracksMeta/' + str(data_id) + '_trackMeta.csv', "r") as tmp_file:
        csvReader = csv.reader(tmp_file)
        header = next(csvReader)
        class_index = header.index("class")
        for row in csvReader:
            class_tmp = row[class_index]
            tracksClass.append(class_tmp)

    return landmark, recordingMeta, tracks, tracksMeta, tracksClass

def coordinate_conversion(tracks, landmark, recordingMeta, origin_GT):
    meter_per_pixel = recordingMeta[15]
    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    landmark1_GT = origin_GT[0]
    landmark2_GT = origin_GT[1]
    landmark3_GT = origin_GT[2]
    landmark4_GT = origin_GT[3]
    b = np.array([landmark1_GT, landmark2_GT, landmark3_GT, landmark4_GT])

    for i in range(len(landmark)):
        cur_frame = landmark[i,1]
        landmark1 = [landmark[i, 2] * meter_per_pixel, -landmark[i, 3] * meter_per_pixel]
        landmark2 = [landmark[i, 4] * meter_per_pixel, -landmark[i, 5] * meter_per_pixel]
        landmark3 = [landmark[i, 6] * meter_per_pixel, -landmark[i, 7] * meter_per_pixel]
        landmark4 = [landmark[i, 8] * meter_per_pixel, -landmark[i, 9] * meter_per_pixel]
        a = np.array([landmark1,landmark2, landmark3, landmark4])
        conversion = np.linalg.lstsq(a,b)[0]
        new_tracks[new_tracks[:, 2] == cur_frame, 4:6] = np.matmul(new_tracks[new_tracks[:, 2] == cur_frame, 4:6],conversion)

    return new_tracks

import os
import numpy as np
import csv
import sys
import time
sys.path.extend(['/home/jhs/Desktop/data_driven_scenario_gen/'])
import lcm
from lcm_def.morai_tx import xsim_vehicle_global_info


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

landmark, recordingMeta, tracks, tracksMeta, tracksClass = data_load(selected_scenario_id)
origin_GT = [[landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 2] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 3] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 4] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 5] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 6] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 7] * recordingMeta[15]],
             [landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 8] * recordingMeta[15], -landmark[np.where(landmark[:,1]==recordingMeta[3])[0][0], 9] * recordingMeta[15]]]
new_tracks = coordinate_conversion(tracks, landmark, recordingMeta, origin_GT)

init_time = time.time() * 10**9
timer_origin = init_time
timer = 0
fps = 29.97
vehicle_state_lcm = lcm.LCM()
vehicle_state = xsim_vehicle_global_info()
cur_frame = -1
while True:
    timer = time.time() * 10**9  - timer_origin
    if timer > 1/fps * 10**9:
        cur_frame = cur_frame + 1
        timer_origin = time.time() * 10**9

        if np.sum(new_tracks[:, 2] == cur_frame) > 0:
            vehicle_state.ntime = int(time.time()*10**9 - init_time)
            vehicle_state.num_of_vehicle = int(np.sum(new_tracks[:, 2] == cur_frame))
            vehicle_state.TV_mark = np.zeros(vehicle_state.num_of_vehicle, dtype = int)
            vehicle_state.id = new_tracks[new_tracks[:, 2] == cur_frame,1].astype(int)
            vehicle_state.x_pos = new_tracks[new_tracks[:, 2] == cur_frame,4]
            vehicle_state.y_pos = new_tracks[new_tracks[:, 2] == cur_frame,5]
            vehicle_state.x_vel = new_tracks[new_tracks[:, 2] == cur_frame, 9]
            vehicle_state.y_vel = new_tracks[new_tracks[:, 2] == cur_frame, 10]
            vehicle_state.length = new_tracks[new_tracks[:, 2] == cur_frame, 8]
            vehicle_state.width = new_tracks[new_tracks[:, 2] == cur_frame, 7]
            vehicle_state.heading = new_tracks[new_tracks[:, 2] == cur_frame, 6]
            vehicle_state.lane_id = np.zeros(vehicle_state.num_of_vehicle, dtype = int)
            vehicle_state.dist_to_left = np.zeros(vehicle_state.num_of_vehicle)
            vehicle_state.dist_to_right = np.zeros(vehicle_state.num_of_vehicle)
            vehicle_state_lcm.publish("test",vehicle_state.encode())
            print('LCM message is published', 'frame : '+str(cur_frame))

