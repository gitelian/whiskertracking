#!/bin/bash
## Takes output from DeepLabCut and computes whisker angles for all experiments
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from moviepy.editor import VideoFileClip


main_dir = '/home/greg/GT015_LT_hsv/'
OVERWRITE = True
#NUM_RAND_FRAMES = 5 # number of frames the user will use to mark the follicle base

# REMEMBER glob.glob returns a LIST of path strings. You must
# index into the appropriate one for whatever experiment you want to analyze
exp_list = np.sort(glob.glob(main_dir + 'FID*'))
num_exps = len(exp_list)
follicle_arrays = list()


## Mark follicle positions ##
# Open a video file for each unique experiment. Randomly open n frames and have
# the user mark the whisker follical for each tracked whisker. Indicate which
# whisker the user must be marking!

for k, exp_n in enumerate(exp_list):

#    SELECT_POINTS = 1

#    while SELECT_POINTS:
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
    for follicle_ind in range(num_whiskers):
        plt.plot(all_coords[:, 0, follicle_ind], all_coords[:, 1, follicle_ind], '.', markersize=5)

    plt.title('{}\nClick on {} then middle click to submit points'.format(first_file[0], whisker_base_keys))
    coords = plt.ginput(-1, show_clicks=True)

    follicle_arrays.append(np.asarray(coords))

        ## select random frames, user chooses follicle positions, all follicle
        #  positions are averaged and presented back to user, user chooses
        #  whether to redo the selection process.

#        # preallocate array to contain the follicle coordinates
#        follicle_coords = np.zeros((num_whiskers, 2, NUM_RAND_FRAMES))
#        rand_frame_indices = np.random.randint(low=0, high=num_frames, size=NUM_RAND_FRAMES)
#
#
#        for frame_count, frame_index in enumerate(rand_frame_indices):
#
#            row = df.loc[frame_index, :]
#
#            # show frame and have user click on follicle positions
#            plt.figure()
#            plt.imshow(mov.get_frame(frame_index / mov.fps))
#            for whisker in whisker_base_keys:
#                plt.plot(row[whisker]['x'], row[whisker]['y'], '-o')
#            # plot tracked points
#            plt.title('{}\nClick on {} then middle click to submit points'.format(first_file[0], whisker_base_keys))
#            coords = plt.ginput(-1, show_clicks=True)
#            plt.close()
#            follicle_coords[:, :, frame_count] = np.asarray(coords)[0:num_whiskers, :] # populates columnwise therefore must transpose first
#
#        follicle_pos = np.mean(follicle_coords, axis=2)
#
#        plt.figure()
#        plt.imshow(mov.get_frame(0))
#        for follicle_ind in range(num_whiskers):
#            plt.plot(follicle_coords[follicle_ind, 0, :], follicle_coords[follicle_ind, 1, :], '.', markersize=5)
#            plt.plot(follicle_pos[follicle_ind, 0], follicle_pos[follicle_ind, 1], '-ro', markerfacecolor="none")
#        plt.title('Click anywhere to close figure')
#        plt.ginput(1)
#        plt.close()
#
#        SELECT_POINTS = input("Re-select points? 0/1: ")



    del df, mov
















































