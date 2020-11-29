#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 04:18:25 2020

@author: jhs
"""

def data_list_load():
    file_list = os.listdir('../mod_data/landmark')
    file_list_int = np.zeros(len(file_list), dtype=int)
    for i in range(len(file_list)):
        file_list_int[i] = int(file_list[i][0:4])

    return file_list_int


def data_load(data_id):
    # background = mpimg.imread('../data/background/' + str(data_id) + '.jpg')
    landmark = np.genfromtxt('../mod_data/landmark/' + str(data_id) +'_landmarks.csv', skip_header=1, delimiter=',',dtype = int)
    landmark[:,1] = landmark[:,1] % 10000
    recordingMeta = np.genfromtxt('../mod_data/recordingMeta/' + str(data_id) + '_recordingMeta.csv', skip_header=1, delimiter = ',')
    recordingMeta[3] = recordingMeta[3] % 10000
    tracks = np.genfromtxt('../mod_data/tracks/' + str(data_id) + '_tracks.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.genfromtxt('../mod_data/tracksMeta/' + str(data_id) + '_trackMeta.csv', skip_header=1, delimiter = ',')
    tracksMeta = np.delete(tracksMeta, -1, -1)
    tracksClass = []
    with open('../mod_data/tracksMeta/' + str(data_id) + '_trackMeta.csv', "r") as tmp_file:
        csvReader = csv.reader(tmp_file)
        header = next(csvReader)
        class_index = header.index("class")
        for row in csvReader:
            class_tmp = row[class_index]
            tracksClass.append(class_tmp)

    return landmark, recordingMeta, tracks, tracksMeta, tracksClass

def coordinate_conversion(tracks, landmark, recordingMeta, origin_GT):
    global center_GT
    global landmark1_GT
    global landmark2_GT
    global landmark3_GT
    global landmark1
    global landmark2
    global landmark3

    meter_per_pixel = recordingMeta[15]
    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    landmark1_GT = np.asarray([origin_GT[0]])
    landmark2_GT = np.asarray([origin_GT[1]])
    landmark3_GT = np.asarray([origin_GT[2]])
    center_GT = [(landmark1_GT[0, 0] + landmark2_GT[0, 0] + landmark3_GT[0, 0]) / 3, (landmark1_GT[0, 1] + landmark2_GT[0, 1] + landmark3_GT[0, 1]) / 3]

    for i in range(len(landmark)):
        print(i)
        cur_frame = landmark[i,1]
        landmark1 = np.asarray([[landmark[i, 2] * meter_per_pixel, -landmark[i, 3] * meter_per_pixel]])
        landmark2 = np.asarray([[landmark[i, 4] * meter_per_pixel, -landmark[i, 5] * meter_per_pixel]])
        landmark3 = np.asarray([[landmark[i, 6] * meter_per_pixel, -landmark[i, 7] * meter_per_pixel]])
        center = [(landmark1[0, 0] + landmark2[0, 0] + landmark3[0, 0]) / 3, (landmark1[0, 1] + landmark2[0, 1] + landmark3[0, 1]) / 3]

        res = minimize(f, [center_GT[0] - center[0], center_GT[1] - center[1], 0], method='Nelder-Mead', tol=1e-10)

        trans_x = res.x[0]
        trans_y = res.x[1]
        rot = res.x[2]

        veh_list = np.where(tracks[:,2]==cur_frame)[0]
        for j in range(len(veh_list)):
            cur_pos = np.asarray([tracks[veh_list[j], 4:6]])
            theta_1 = np.rad2deg(np.arctan2(cur_pos[0][1], cur_pos[0][0]))

            x_1 = trans_x + np.sqrt(cur_pos[0][0] ** 2 + cur_pos[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_1))
            y_1 = trans_y + np.sqrt(cur_pos[0][0] ** 2 + cur_pos[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_1))

            new_tracks[veh_list[j], 4:6] = np.asarray([x_1, y_1])
            new_tracks[veh_list[j], 6] = new_tracks[veh_list[j], 6] + rot + 90

    return new_tracks

def f(x):
    trans_x = x[0]
    trans_y = x[1]
    rot = x[2]

    theta_1 = np.rad2deg(np.arctan2(landmark1[0][1], landmark1[0][0]))
    theta_2 = np.rad2deg(np.arctan2(landmark2[0][1], landmark2[0][0]))
    theta_3 = np.rad2deg(np.arctan2(landmark3[0][1], landmark3[0][0]))

    x_1 = trans_x + np.sqrt(landmark1[0][0]**2 + landmark1[0][1]**2) * np.cos(np.deg2rad(rot + theta_1))
    y_1 = trans_y + np.sqrt(landmark1[0][0]**2 + landmark1[0][1]**2) * np.sin(np.deg2rad(rot + theta_1))

    x_2 = trans_x + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_2))
    y_2 = trans_y + np.sqrt(landmark2[0][0] ** 2 + landmark2[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_2))

    x_3 = trans_x + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.cos(np.deg2rad(rot + theta_3))
    y_3 = trans_y + np.sqrt(landmark3[0][0] ** 2 + landmark3[0][1] ** 2) * np.sin(np.deg2rad(rot + theta_3))

    landmark1_trans = np.asarray([[x_1, y_1]])
    landmark2_trans = np.asarray([[x_2, y_2]])
    landmark3_trans = np.asarray([[x_3, y_3]])

    return np.linalg.norm(landmark1_GT - landmark1_trans) + np.linalg.norm(landmark2_GT - landmark2_trans) + np.linalg.norm(landmark3_GT - landmark3_trans)

import os
import numpy as np
import csv
import sys
import time
sys.path.extend(['/home/jhs/Desktop/data_driven_scenario_gen/'])
import lcm
from lcm_def.morai_tx import xsim_vehicle_global_info
from lcm_def.morai_tx import xsim_ego_info
from scipy.optimize import minimize, rosen, rosen_der

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
origin_GT = [[641.484, -1080.898],
             [653.099, -1110.089],
             [629.438, -1119.350]]

new_tracks = coordinate_conversion(tracks, landmark, recordingMeta, origin_GT)

init_time = time.time() * 10**9
timer_origin = init_time
timer = 0
fps = 29.97
vehicle_state_lcm = lcm.LCM()
ego_state_lcm = lcm.LCM()
vehicle_state = xsim_vehicle_global_info()
ego_state = xsim_ego_info()

ego_state.x_pos_ego = 0
ego_state.y_pos_ego = 0
ego_state.heading_ego = 0
ego_state.blinker_info = int(0)
ego_state.steering_angle = 0
ego_state.fl_wheel_vel = 0
ego_state.fr_wheel_vel = 0
ego_state.rl_wheel_vel = 0
ego_state.rr_wheel_vel = 0

cur_frame = -1
while True:
    timer = time.time() * 10**9  - timer_origin
    if timer > 1/fps * 10**9:
        cur_frame = cur_frame + 1
        timer_origin = time.time() * 10**9

        if np.sum(new_tracks[:, 2] == cur_frame) > 0:
            vehicle_state.ntime = int(time.time()*10**9 - init_time)
            ego_state.ntime = int(time.time()*10**9 - init_time)
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
            vehicle_state_lcm.publish("MORAI_XSIM_VEHICLE_INFO",vehicle_state.encode())
            ego_state_lcm.publish("MORAI_EGO_INFO",ego_state.encode())
            print('LCM message is published', 'frame : '+str(cur_frame))
