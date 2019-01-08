#!/bin/bash
## Takes output from DeepLabCut and computes whisker angles for all experiments
import os
import glob
import pandas as pd


main_dir = '/home/greg/GT015_LT_hsv/'
OVERWRITE = True

# REMEMBER glob.glob returns a LIST of path strings. You must
# index into the appropriate one for whatever experiment you want to analyze
exp_list = sort(glob.glob(main_dir + 'FID*'))
num_exps = len(exp_list)


## Mark follicle positions ##
# Open a video file for each unique experiment. Randomly open n frames and have
# the user mark the whisker follical for each tracked whisker. Indicate which
# whisker the user must be marking!

for exp_n in exp_list:
    # sort videos and get first video name
    avi_list = sort(glob.glob(exp_n + os.sep + '*.avi'))
    first_file = os.path.splitext(os.path.basename(avi_list[0]))
    first_h5_file = glob.glob(exp_n + os.sep + first_file[0] + '*.h5')

    if len(first_h5_file) != 1:
        raise Exception('No .h5 file found for {} rerun deeplabcut.analyze'.format(first_h5_file))

    # open h5 file with pandas
    df = pd.read_hdf(first_h5_file[0])

    ## get list of unique bodyparts/things labeled. Points on the same whisker will
    #  have their angle computed from the same follicle

    # returns an array of size (n,) where n is the number of measurements such
    # as the x, y coordinates as well as the likelihood of the tracked point
    # being present in the frame. Each entry in the list is a type of size 3.

    column_values = df.columns.values
    entry_list = [list() for x in range(len(column_values[0]))]

    for k, entry in enumerate(column_values):
        entry_list[0].append(column_values[k][0])
        entry_list[1].append(column_values[k][1])
        entry_list[2].append(column_values[k][2])

    scorer = [str(x) for x in np.unique(entry_list[0])[:]]
    bodyparts = [str(x) for x in np.unique(entry_list[1])[:]]
    measurement_type = [str(x) for x in np.unique(entry_list[2])[:]]


