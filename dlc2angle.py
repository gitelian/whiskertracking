#!/bin/bash
## Takes output from DeepLabCut and computes whisker angles for all experiments
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from moviepy.editor import VideoFileClip
import multiprocessing as mp
import time


main_dir = '/home/greg/GT015_LT_hsv/'
OVERWRITE = True

# REMEMBER glob.glob returns a LIST of path strings. You must
# index into the appropriate one for whatever experiment you want to analyze
exp_list = np.sort(glob.glob(main_dir + 'FID*'))
num_exps = len(exp_list)
follicle_arrays = list()


## Mark follicle positions ##
# Open a video file for each unique experiment. Have the user mark the whisker
# follical for each tracked whisker. Indicate which whisker the user must be marking!

for k, exp_n in enumerate(exp_list):

    # sort videos and get first video name
    avi_list = np.sort(glob.glob(exp_n + os.sep + '*.avi'))
    first_file = os.path.splitext(os.path.basename(avi_list[0]))
    first_h5_file = glob.glob(exp_n + os.sep + first_file[0] + '*.h5')

    if len(first_h5_file) != 1:
        raise Exception('No .h5 file found for {} rerun deeplabcut.analyze'.format(first_h5_file))

    # open h5 file with pandas
    df = pd.read_hdf(first_h5_file[0])

    # get rid of scorer level!
    df.columns = df.columns.droplevel()
    row = df.loc[0, :]
    bodyparts = [str(x) for x in row.unstack(level=0).keys().tolist()]

    # get unique whisker_base keys
    whisker_base_keys = np.sort([x for x in bodyparts if "base" in x])
    num_whiskers = len(whisker_base_keys)
    num_frames = df.shape[0]

    # open movie file
    mov = VideoFileClip(avi_list[k])

    ## plot all whisker base points and have user select follicle positions
    all_coords = np.zeros((num_frames, 2, num_whiskers))
    for frame_index in range(num_frames):

        row = df.loc[frame_index, :]

        for whisker_ind, whisker in enumerate(whisker_base_keys):
            all_coords[frame_index, :, whisker_ind] = np.asarray([row[whisker]['x'], row[whisker]['y']])

    plt.imshow(mov.get_frame(0))
    row = df.loc[0, :]
    for follicle_ind, whisker in enumerate(whisker_base_keys):
        plt.plot(all_coords[:, 0, follicle_ind], all_coords[:, 1, follicle_ind], '.', markersize=5)
        plt.plot(row[whisker]['x'], row[whisker]['y'], '-ro', markerfacecolor='none')

    plt.title('{}\nClick on {} then middle click to submit points'.format(first_file[0], whisker_base_keys))
    coords = plt.ginput(-1, show_clicks=True)
    plt.close()

    follicle_arrays.append(np.asarray(coords))

    del df, mov

##### Functions for data extraction #####
##### Functions for data extraction #####

def extract_data(h5, whisker_base_keys, object_key, order):

    # open h5 file with pandas
    df = pd.read_hdf(h5)

    # get rid of scorer level!
    df.columns = df.columns.droplevel()

    num_frames = df.shape[0]
    angles_temp = np.zeros((num_frames, len(whisker_base_keys)))
    object_pos_temp = np.zeros((num_frames, 2))

    for frame_ind in range(num_frames):
        row = df.loc[frame_ind, :]

        for whisker_ind, whisker_key in enumerate(whisker_base_keys):
            # if whisker likelihood if high do the rest
            if row[whisker_key]['likelihood'] > 0.95:
                x1 = row[whisker_key]['x']
                y1 = row[whisker_key]['y']
                x0, y0 = follicle_arrays[k][whisker_ind, :]

                if x0 > x1:
                    x = x0 - x1
                    y = y0 - y1

                    # compute angle
                    ang = 180 - np.rad2deg(np.arctan(y/x))

                elif x0 < x1:
                    x = x1 - x0
                    y = y0 - y1

                    # compute angle
                    ang = 180 - np.rad2deg(np.arctan(y/x))

                # append to angles array
                angles_temp[frame_ind, whisker_ind] = ang

        if object_key:
            # if object present append coordinates
            if row[object_key[0]]['likelihood'] > 0.95:
                object_pos_temp[frame_ind, 0] = row[object_key[0]]['x']
                object_pos_temp[frame_ind, 1] = row[object_key[0]]['y']

    return order, angles_temp, object_pos_temp



##### END functions for data extraction #####
##### END functions for data extraction #####



print('COMPUTING ANGLE FOR ALL EXPERIMENTS')

for k, exp_n in enumerate(exp_list):

    # sort videos and get first video name
    h5_list = np.sort(glob.glob(exp_n + os.sep + '*.h5'))

    if len(first_h5_file) != 1:
        raise Exception('No .h5 file found for {} rerun deeplabcut.analyze'.format(first_h5_file))

    # open h5 file with pandas
    df = pd.read_hdf(h5_list[k])

    # get rid of scorer level!
    df.columns = df.columns.droplevel()
    row = df.loc[0, :]
    bodyparts = [str(x) for x in row.unstack(level=0).keys().tolist()]

    # get unique whisker_base keys
    whisker_base_keys = np.sort([x for x in bodyparts if "base" in x])
    num_whiskers = len(whisker_base_keys)
    num_frames = df.shape[0]
    num_files = len(h5_list)

    # get object key
    object_key = [x for x in bodyparts if "object" in x]
    object_pos = np.zeros((num_frames, 2, num_files))

    # pre-allocate
    angles = np.zeros((num_frames, num_whiskers, num_files))

    #for file_ind, h5 in enumerate(h5_list):
    #    angles_temp, object_pos_temp = extract_data(h5, whisker_base_keys, object_key)
    #    angles[:, :, file_ind] = angles_temp
    #    object_pos[:, :, file_ind] = object_pos_temp

    # setup parallel pool
    processes = 4
    pool = mp.Pool(processes)
    t = time.time()

    # run parallel processes
    results = [pool.apply_async(extract_data, args=(h5, whisker_base_keys, object_key, i)) for i, h5 in enumerate(h5_list)]
    results = [p.get() for p in results]

    # ensure output is in correct order. 'apply_async' does not ensure order preservation
    order, data = zip(*[(entry[0],entry[1]) for entry in results])
    sort_ind = np.argsort(order)
    for ind in sort_ind:
        angles[:, :, ind] = data[0]
        object_pos[:, :, ind] = data[1]

    elapsed = time.time() - t
    pool.close()
    print('total time: ' + str(elapsed))







    # convert to pandas data frame and then convert to CSV for reading in to
    # MATLAB
#    df = pd.DataFrame(angles, columns=whisker_base_keys)












































