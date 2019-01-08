#!/bin/bash
## Takes output from DeepLabCut and computes whisker angles for all experiments
import os
import glob
import pandas as pd
from moviepy.editor import VideoFileClip


main_dir = '/home/greg/GT015_LT_hsv/'
OVERWRITE = True
NUM_RAND_FRAMES = 2 # number of frames the user will use to mark the follicle base

# REMEMBER glob.glob returns a LIST of path strings. You must
# index into the appropriate one for whatever experiment you want to analyze
exp_list = sort(glob.glob(main_dir + 'FID*'))
num_exps = len(exp_list)


## Mark follicle positions ##
# Open a video file for each unique experiment. Randomly open n frames and have
# the user mark the whisker follical for each tracked whisker. Indicate which
# whisker the user must be marking!

for k, exp_n in enumerate(exp_list):
    # sort videos and get first video name
    avi_list = sort(glob.glob(exp_n + os.sep + '*.avi'))
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
    whisker_base_keys = sort([x for x in bodyparts if "base" in x])
    num_whiskers = len(whisker_base_keys)
    num_frames = df.shape[0]

    # preallocate array to contain the follicle coordinates
    follicle_coords = np.zeros((num_whiskers, 2, NUM_RAND_FRAMES))
    rand_frame_indices = np.random.randint(low=0, high=num_frames, size=NUM_RAND_FRAMES)

    # open movie file
    mov = VideoFileClip(avi_list[k])

    for frame_count, frame_index in enumerate(rand_frame_indices):

        row = df.loc[frame_index, :]

        # show frame and have user click on follicle positions
        plt.figure()
        plt.imshow(mov.get_frame(frame_index / mov.fps))
        for whisker in whisker_base_keys:
            plt.plot(row[whisker]['x'], row[whisker]['y'], '-o')
        # plot tracked points
        plt.title('Click on {} then middle click to submit points'.format(whisker_base_keys))
        coords = plt.ginput(-1, show_clicks=True)
        plt.close()
        follicle_coords[:, :, frame_count] = np.asarray(coords) # populates columnwise therefore must transpose first

    follicle_pos = np.mean(follicle_coords, axis=2)



    df['object_center']['x']

