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
    # background = mpimg.imread('../data/background/' + str(data_id) + '.jpg')
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

    return landmark, recordingMeta, tracks, tracksMeta, tracksClass

def coordinate_conversion(tracks, landmark, recordingMeta, origin_GT):
    global center_GT
    global landmark1_GT
    global landmark2_GT
    global landmark3_GT
    global landmark1_trans
    global landmark2_trans
    global landmark3_trans
    global transition_matrix
    
    reference_frame = recordingMeta[3]
    meter_per_pixel_hand = recordingMeta[15]
    dist_1_2 = np.linalg.norm(origin_GT[0,:] - origin_GT[1,:])
    dist_2_3 = np.linalg.norm(origin_GT[1,:] - origin_GT[2,:])
    dist_3_1 = np.linalg.norm(origin_GT[2,:] - origin_GT[0,:])
    
    pixel_1_2_ref_frame = np.linalg.norm(recordingMeta[16:18] - recordingMeta[18:20])
    pixel_2_3_ref_frame = np.linalg.norm(recordingMeta[18:20] - recordingMeta[20:22])
    pixel_3_1_ref_frame = np.linalg.norm(recordingMeta[20:22] - recordingMeta[16:18])
    
    meter_per_pixel_1_2 = dist_1_2/pixel_1_2_ref_frame
    meter_per_pixel_2_3 = dist_2_3/pixel_2_3_ref_frame
    meter_per_pixel_3_1 = dist_3_1/pixel_3_1_ref_frame
    
    meter_per_pixel_gt = np.mean((meter_per_pixel_1_2,meter_per_pixel_2_3,meter_per_pixel_3_1))
    recordingMeta[15] = meter_per_pixel_gt

    new_tracks = np.zeros_like(tracks)
    new_tracks[:] = tracks[:]
    new_tracks[:,4:6] = (new_tracks[:,4:6]  / meter_per_pixel_hand) * meter_per_pixel_gt


    return new_tracks, recordingMeta


import os
import numpy as np
import csv
import sys

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
origin_GT = np.asarray([[641.484, -1080.898],
	     [653.099, -1110.089],
             [629.438, -1119.350]])

new_tracks, new_recordingMeta = coordinate_conversion(tracks, landmark, recordingMeta, origin_GT)
new_recordingMeta = np.asarray([new_recordingMeta])
f = open('tracks/'+str(selected_scenario_id)+'_tracks.csv','w')
wr = csv.writer(f)
wr.writerow(['recordingId','trackId','frame','trackLifetime','xCenter','yCenter','heading','width','length','XVelocity','YVelocity','Acceleration','yAcceleration','lonVelocity','latVelocity','lonAcceleration','latAcceleration'])
for row in new_tracks:
    wr.writerow(row)
f.close()

f = open('recordingMeta/'+str(selected_scenario_id)+'_recordingMeta.csv','w')
wr = csv.writer(f)
wr.writerow(['recordingId','trackId','frameRate','referenceFrame','weekday','startTime','duration','numTracks','numVehicles','numVRUs','latLocation','lonLocation','xUtmOrigin','yUtmOrigin','orthoPxToMeter','px2meter','p1x','p1y','p2x','p2y','p3x','p3y','p4x','p4y'])
for row in new_recordingMeta:
    wr.writerow(row)
f.close()



