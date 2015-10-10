#!/bin/bash
from neuroanalyzer import *
from corrstats import *
import seaborn as sns
import statsmodels.stats.multitest as smm

def get_wt_run_vals(wtmat, trials_ran_dict, cond_ind_dict, vel_mat, trial_time):
    num_trials_ran = [sum(trials_ran_dict[x]) for x in np.sort(trials_ran_dict.keys())]
    num_frames = wtmat['numFrames'][0]
    camTime = wtmat['camTime'][:, 0]
    set_point_list = list()
    amp_list = list()
    num_samples = vel_mat.shape[1]
    vel_list = list()
    ang_list = list()
    for cond_ind, cond in enumerate(np.sort(trials_ran_dict.keys())):
        print(cond)
        temp_setpoint = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_amp = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_ang = np.zeros((num_trials_ran[cond_ind], num_frames))
        temp_vel = np.zeros((num_trials_ran[cond_ind], num_samples))
        temp_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
        run_inds  = cond_ind_dict[cond][temp_inds]
        print(run_inds)

        # Given a condition iterate through all of its running trials
        good_inds = list()
        for trial_ind,runind in enumerate(run_inds):

            temp_sp = wtmat['setPointMat'][runind]
            temp_a  = wtmat['angleMat'][runind]
            bad_set_point = np.sum(np.logical_or(temp_sp < 0, temp_sp > 200)) > 0
            bad_angle = np.sum(np.logical_or(temp_a < 0, temp_a > 200)) > 0
            if not bad_set_point and not bad_angle:
                good_inds.append(trial_ind)

            temp_setpoint[trial_ind, :] = wtmat['setPointMat'][runind]
            temp_amp[trial_ind, :] = wtmat['ampMat'][runind]
            temp_ang[trial_ind, :] = wtmat['angleMat'][runind]
            temp_vel[trial_ind, :] = vel_mat[runind, :]
        set_point_list.append(temp_setpoint[good_inds, :])
        amp_list.append(temp_amp[good_inds, :])
        vel_list.append(temp_vel[good_inds, :])
        ang_list.append(temp_ang[good_inds, :])

    return camTime, set_point_list, amp_list, vel_list, ang_list

def velocity_analysis_parameters(wtmat, trials_ran_dict, cond_ind_dict, vel_mat, trial_time):
    numframes2bin = 50
    num_trials_ran = [sum(trials_ran_dict[x]) for x in np.sort(trials_ran_dict.keys())]
    num_frames = wtmat['numFrames'][0]
    camTime = wtmat['camTime'][:, 0]
    run_time_ind = [np.argmin(np.abs(x - trial_time)) for x in camTime]
    set_point_list = list()
    amp_list = list()
    num_samples = vel_mat.shape[1]
    vel_list = list()
    for cond_ind, cond in enumerate(np.sort(trials_ran_dict.keys())):
        print(cond)
        temp_setpoint = np.empty(0)
        temp_amp = np.empty(0)
        temp_vel = np.empty(0)
        #temp_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]
        run_inds  = cond_ind_dict[cond]#[temp_inds]
        print(run_inds)

        # Given a condition iterate through all of its running trials
        for trial_ind,runind in enumerate(run_inds):
            temp_setpoint = np.concatenate((temp_setpoint,
                np.nanmean(wtmat['setPointMat'][runind].reshape(-1, numframes2bin), axis=1)))
            temp_amp = np.concatenate((temp_amp,
                np.nanmean(wtmat['ampMat'][runind].reshape(-1, numframes2bin), axis=1)))
            temp_vel = np.concatenate((temp_vel,
                np.nanmean(vel_mat[runind, run_time_ind].reshape(-1, numframes2bin), axis=1)))
        good_inds = np.where(np.logical_and(temp_setpoint >= 0, temp_setpoint <= 200))
        set_point_list.append(temp_setpoint[good_inds])
        amp_list.append(temp_amp[good_inds])
        vel_list.append(temp_vel[good_inds]) # removes columns

    return camTime, set_point_list, amp_list, vel_list

def plot_wt_traces(set_point_list, amp_list):

    positions = 9
    f1, ax1 = plt.subplots()

    for x in range(positions):
        set_mean_pretrim = np.nanmean(set_point_list[x], axis=0)
        set_std_pretrim  = np.nanstd(set_point_list[x], axis=0)
        set_mean_posttrim = np.nanmean(set_point_list[x+9], axis=0)
        set_std_posttrim  = np.nanstd(set_point_list[x+9], axis=0)

        plt.subplot(2, 5, x+1)
        plt.plot(camTime, set_mean_pretrim, 'b', wtmat['camTime'], set_mean_posttrim, 'r')
        plt.fill_between(camTime, set_mean_pretrim + set_std_pretrim,
                set_mean_pretrim - set_std_pretrim, facecolor='b', alpha=0.3)
        plt.fill_between(camTime, set_mean_posttrim + set_std_posttrim,
                set_mean_posttrim - set_std_posttrim, facecolor='r', alpha=0.3)
        plt.plot(camTime,np.transpose(set_point_list[x]), 'b', alpha=0.7)
        plt.plot(camTime,np.transpose(set_point_list[x+9]), 'r', alpha=0.7)
        plt.ylim(80, 170)

    f2, ax2 = plt.subplots()
    for x in range(positions):
        plt.ylim(120, 160)

        amp_mean_pretrim  = np.nanmean(amp_list[x], axis=0)
        amp_std_pretrim   = np.nanstd(amp_list[x], axis=0)/np.sqrt(amp_list[x].shape[0])
        amp_mean_posttrim = np.nanmean(amp_list[x+9], axis=0)
        amp_std_posttrim  = np.nanstd(amp_list[x+9], axis=0)/np.sqrt(amp_list[x+9].shape[0])

        plt.subplot(2, positions, x+1+positions)
        plt.plot(camTime, amp_mean_pretrim, 'b', wtmat['camTime'][0], amp_mean_posttrim, 'r')
        plt.fill_between(camTime, amp_mean_pretrim + amp_std_pretrim,
                amp_mean_pretrim - amp_std_pretrim, facecolor='b', alpha=0.5)
        plt.fill_between(camTime, amp_mean_posttrim + amp_std_posttrim,
                amp_mean_posttrim - amp_std_posttrim, facecolor='r', alpha=0.5)
        plt.ylim(0, 25)
    plt.show()

    return f1, ax1, f2, ax2

def get_cummulative_distributions(wtmat, trials_ran_dict, trial_time, start_time=1.5, stop_time=2.5, control_pos=9):
    '''
    Compute cummulative whisker parameters during analysis window.
    '''

    camTime = wtmat['camTime'][:, 0]
    num_frames = wtmat['numFrames'][0]

    conditions = np.sort(trials_ran_dict.keys())
    num_trials_ran = [sum(trials_ran_dict[x]) for x in conditions]

    set_point_cum = list()
    amp_cum       = list()
    angle_cum     = list()
    total_trials  = list()
    run_cum       = list()

    camInds = np.logical_and(camTime >= start_time, camTime <= stop_time)
    camIndsControl = range(num_frames)
    velRunInds = np.logical_and(trial_time >= start_time, trial_time <= stop_time)
    velRunIndsControl = np.logical_and(trial_time >= wtmat['startTime'][0], trial_time <= wtmat['stopTime'][0])

    # Iterate through all conditions for a given unit
    for i, cond in enumerate(conditions):
        print('condition ' + str(i) + ' of ' + str(len(conditions)))
        cond_set_point = list()
        cond_amp = list()
        cond_angle = list()

        cum_cond_set_point = np.empty((1,1))
        cum_cond_amp = np.empty((1,1))
        cum_cond_angle = np.empty((1,1))
        cum_cond_run = np.empty((1,1))


        # Get running indices for the given condition
        cond_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]

        # Get trial index for grabbing the right hsv data
        run_inds  = cond_ind_dict[cond][cond_inds]

        # Iterate through all running trials
        for trial_ind, runind in enumerate(run_inds):
            print('running trial ' + str(trial_ind) + ' of ' + str(len(run_inds)))

            # Get whisker values during analysis time for histogram
            # normalization
            if (i+1)%control_pos != 0:
                cum_cond_set_point = np.append(cum_cond_set_point, wtmat['setPointMat'][runind][camInds])
                cum_cond_amp = np.append(cum_cond_amp, wtmat['ampMat'][runind][camInds])
                cum_cond_angle = np.append(cum_cond_angle, wtmat['angleMat'][runind][camInds])

                cum_cond_run = np.append(cum_cond_run, vel_mat[runind,velRunInds])

            elif (i+1)%control_pos == 0:
                cum_cond_set_point = np.append(cum_cond_set_point, wtmat['setPointMat'][runind][camIndsControl])
                cum_cond_amp = np.append(cum_cond_amp, wtmat['ampMat'][runind][camIndsControl])
                cum_cond_angle = np.append(cum_cond_angle, wtmat['angleMat'][runind][camIndsControl])

                cum_cond_run = np.append(cum_cond_run, vel_mat[runind,velRunIndsControl])

        # Add cummulative whisker info to list
        set_point_cum.append(cum_cond_set_point)
        amp_cum.append(cum_cond_amp)
        angle_cum.append(cum_cond_angle)
        total_trials.append(trial_ind+1)
        print(cum_cond_run)
        run_cum.append(cum_cond_run)

    return set_point_cum, amp_cum, angle_cum, run_cum, total_trials

def get_spike_triggered_averages(df, wtmat, trials_ran_dict, trial_time, start_time=1.5, stop_time=2.5, control_pos=9):
    '''
    Compute spike triggered averages for set-point, amplitude, and angle.
    '''

    num_units = df.shape[0]

    conditions = np.sort(trials_ran_dict.keys())
    num_trials_ran = [sum(trials_ran_dict[x]) for x in conditions]

    set_point_sta = list()
    amp_sta = list()
    angle_sta = list()

    camTime = wtmat['camTime'][:]
    num_frames = wtmat['numFrames'][:]

    camInds = np.logical_and(camTime >= start_time, camTime <= stop_time)
    camIndsControl = range(num_frames)
    velRunInds = np.logical_and(trial_time >= start_time, trial_time <= stop_time)
    velRunIndsControl = np.logical_and(trial_time >= wtmat['startTime'][0], trial_time <= wtmat['stopTime'][0])

    # Iterate through all units in data frame
    for unit_ind in range(num_units):
        print('working on unit ' + str(unit_ind))
        unit_set_point = list()
        unit_amp = list()
        unit_angle = list()

        # Iterate through all conditions for a given unit
        for i, cond in enumerate(conditions):
            print('condition ' + str(i) + ' of ' + str(len(conditions)))
            cond_set_point = list()
            cond_amp = list()
            cond_angle = list()

            # Get running indices for the given condition
            cond_inds = np.where(np.array(trials_ran_dict[cond]) == True)[0]

            # Get trial index for grabbing the right hsv data
            run_inds  = cond_ind_dict[cond][cond_inds]

            # Iterate through all running trials
            for trial_ind, runind in enumerate(run_inds):
                print('running trial ' + str(trial_ind) + ' of ' + str(len(run_inds)))

                # Get spike times for analysis window
                st = df[cond][unit_ind][cond_inds[trial_ind]]
                if (i+1)%control_pos != 0:
                    st = st[np.logical_and(st >= start_time, st <= stop_time)]
                # Get spike times for entire window during control positions
                else:
                    st = st[np.logical_and(st >= wtmat['startTime'][0], st <= wtmat['stopTime'][0])]

                # Iterate through all spike times
                for spike_time in st:

                    # Get hsv index closest to spike time
                    hsv_ind = np.argmin(np.abs(camTime - spike_time))

                    # Add hsv data corresponding to that spike time
                    cond_set_point.append(wtmat['setPointMat'][runind][hsv_ind])
                    cond_amp.append(wtmat['ampMat'][runind][hsv_ind])
                    cond_angle.append(wtmat['angleMat'][runind][hsv_ind])


            # Add condition data to unit lists
            unit_set_point.append(cond_set_point)
            unit_amp.append(cond_amp)
            unit_angle.append(cond_angle)

        # Add unit data to sta lists
        set_point_sta.append(unit_set_point)
        amp_sta.append(unit_amp)
        angle_sta.append(unit_angle)

    return set_point_sta, amp_sta, angle_sta

def mean_confidence_interval(data, confidence=0.95):
    '''
    Compute the mean, standard error of the mean and the 95% confidence interval.
    '''
    n = len(data)
    m, se = np.nanmean(data), np.nanstd(data)/np.sqrt(len(data))
    h = se*sp.stats.t.ppf((1+confidence)/2.0, n-1)
    return m, se, h

def cohens_d(x, y):
    '''
    Compute Cohen's D for two independent samples

    Formula's found here:
    http://trendingsideways.com/index.php/cohens-d-formula/
    '''
    m1 = np.nanmean(x)
    m2 = np.nanmean(y)
    n1, n2 = len(x), len(y)

    sd_pool = np.sqrt((np.nansum( (x - np.nanmean(x))**2 ) + np.nansum( (y - np.nanmean(y))**2 ))/
            float(n1 + n2 - 2))

    d = np.abs((m1 - m2)/sd_pool)

    v = ( (n1 + n2)/(n1*n2) + d**2/(2*(n1 + n2 - 2)))*( (n1 + n2)/(n1 + n2 - 2))
    v = v/(n1 + n2 -2)

    return d, v

def bootstrap_medians(x, y, num_samples=10000):

    z = np.concatenate((x, y))
    num_xsamples = x.shape[0]
    num_ysamples = y.shape[0]
    boot_dist    = np.zeros((num_samples, 1))

    for i in range(num_samples):
        x_med = np.median(np.random.choice(z, num_xsamples))
        y_med = np.median(np.random.choice(z, num_ysamples))
        boot_dist[i] = x_med - y_med

    return boot_dist

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    fid = '1067'
    region = 'vS1'

    sns.set_context("poster")

    usr_dir = os.path.expanduser('~')
    sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
    fid_region = 'fid' + fid + '_' + region
    sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

    data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
    data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

    # Calculate runspeed
    run_mat = load_run_file(data_dir_paths[0]).value
    vel_mat, trial_time = calculate_runspeed(run_mat)

    # Get stimulus id list
    stim = load_stimsequence(data_dir_paths[0])

    # Create running trial dictionary
    # Strict running thresholds
    cond_ind_dict,trials_ran_dict = classify_run_trials(stim, vel_mat,
            trial_time, stim_start=1.25, stim_stop=2.50, mean_thresh=400,
            sigma_thresh=150, low_thresh=200, display=True)

    vel_mat = vel_mat*(2*np.pi*6)/360.0

    # Easy running thresholds
#    cond_ind_dict,trials_ran_dict = classify_run_trials(stim, vel_mat,
#        trial_time, stim_start=1.25, stim_stop=2.50, mean_thresh=175,
#        sigma_thresh=150, low_thresh=100, display=True)

    # Put data into a Pandas dataframe
    #df = make_df(sort_file_paths,data_dir_path,region=region)

    # Bin spike data
    #count_mats = bin_data(df,trials_ran_dict, start_time=0, stop_time=2.5, binsize=0.001)
    #plt.figure()
    #plt.imshow(count_mats[4], interpolation='nearest', aspect='auto')

###############################################################################
##################### Analyze Whisker Tracking Data ###########################
###############################################################################
    hsv_mat_path = glob.glob(usr_dir + '/Documents/AdesnikLab/Processed_HSV/FID' + fid + '-data*.mat')[0]
    wtmat = h5py.File(hsv_mat_path)
    # example: this returns the angle values for trial 7 wtmat['angleCell'][6]

##### Make set-point vs runspeed plots #####
##### Make set-point vs runspeed plots #####

    fids = ['1034', '1038', '1044', '1051', '1054', '1058', '1062', '1067']
    region = 'vS1'

    sns.set_context("poster")
    scorr_coefs = np.empty((0, 2))
    test_pvals  = np.empty((0, 1))

    sp_diff   = np.empty((0, 1))
    sp_ttest_pvals = np.empty((0, 1))
    amp_diff   = np.empty((0, 1))
    amp_ttest_pvals = np.empty((0, 1))

    ang_dict = dict()

    for fid in fids:
        usr_dir = os.path.expanduser('~')
        sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
        fid_region = 'fid' + fid + '_' + region
        sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

        data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
        data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

        # Calculate runspeed
        run_mat = load_run_file(data_dir_paths[0]).value
        vel_mat, trial_time = calculate_runspeed(run_mat)

        # Get stimulus id list
        stim = load_stimsequence(data_dir_paths[0])

        # Create running trial dictionary
        # Strict running thresholds
        cond_ind_dict,trials_ran_dict = classify_run_trials(stim, vel_mat,
                trial_time, stim_start=1.25, stim_stop=2.50, mean_thresh=400,
                sigma_thresh=150, low_thresh=200, display=False)

        vel_mat = vel_mat*(2*np.pi*6)/360.0

        hsv_mat_path = glob.glob(usr_dir + '/Documents/AdesnikLab/Processed_HSV/FID' + fid + '-data*.mat')[0]
        wtmat = h5py.File(hsv_mat_path)


        [camTime, set_point_vals, amp_vals, vel_vals] = velocity_analysis_parameters(wtmat,
                trials_ran_dict, cond_ind_dict, vel_mat, trial_time)

        camTime, set_point_list, amp_list, vel_list, ang_list = get_wt_run_vals(wtmat, trials_ran_dict,
                cond_ind_dict, vel_mat, trial_time)

        ang_dict[fid] = ang_list

        sp_samples_list  = [np.nanmean(x[:, 750:1000], axis=1) for x in set_point_list]
        sp_samples_list  = [np.delete(x, np.where(np.isnan(x))) for x in sp_samples_list] # remove nans

        amp_samples_list  = [np.nanmean(x[:, 750:1000], axis=1) for x in amp_list]
        amp_samples_list  = [np.delete(x, np.where(np.isnan(x))) for x in amp_samples_list] # remove nans

        for i in range(9):

            ##### Compute Spearmans correlation coefficient on set-point vs
            ##### runspeed. Calculated if pre vs post trimming coefficients are
            ##### statistically different using Fisher's z-transform
            pre_slow_inds = np.where(vel_vals[i] < 40)
            pre_vel = np.delete(vel_vals[i], pre_slow_inds)
            pre_sp  = np.delete(set_point_vals[i], pre_slow_inds)

            post_slow_inds = np.where(vel_vals[i+9] < 40)
            post_vel = np.delete(vel_vals[i+9], post_slow_inds)
            post_sp  = np.delete(set_point_vals[i+9], post_slow_inds)

            pre_scorr  = sp.stats.spearmanr(pre_vel, pre_sp)
            post_scorr = sp.stats.spearmanr(post_vel, post_sp)

            scorr_coefs = np.concatenate((scorr_coefs,
                np.asarray((pre_scorr[0], post_scorr[0])).reshape(1,2)), axis=0)
            test_pvals  = np.concatenate((test_pvals,
                independent_corr(pre_scorr[0], post_scorr[0], pre_sp.shape[0], post_sp.shape[0])[1].reshape(1,1)), axis=0)

            ##### Compute ttest on mean set-point measurements and difference
            ##### in means for the stimulus analysis window.
            sp_diff   = np.concatenate((sp_diff,
                np.diff((np.nanmean(sp_samples_list[i]), np.nanmean(sp_samples_list[i+9]))).reshape(1,1)), axis=0)
            sp_ttest_pvals = np.concatenate((sp_ttest_pvals,
                sp.stats.ttest_ind(sp_samples_list[i], sp_samples_list[i+9])[1].reshape(1,1)), axis=0)

            ##### Compute ttest on mean amplitude measurements and difference
            ##### in means for the stimulus analysis window.
            amp_diff   = np.concatenate((amp_diff,
                np.diff((np.nanmean(amp_samples_list[i]), np.nanmean(amp_samples_list[i+9]))).reshape(1,1)), axis=0)
            amp_ttest_pvals = np.concatenate((amp_ttest_pvals,
                sp.stats.ttest_ind(amp_samples_list[i], amp_samples_list[i+9])[1].reshape(1,1)), axis=0)

    rej, pval_corr = smm.multipletests(test_pvals.reshape(test_pvals.shape[0],), alpha=0.05, method='simes-hochberg')[:2]
    print(float(sum(rej))/len(rej))
    corr_diff = np.diff(scorr_coefs, axis=1)
    a = np.concatenate((np.ones((72,1)), np.ones((72,1))*2),axis=1)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(a.T, scorr_coefs.T, '-o')
    plt.xlim(0.5, 2.5)
    plt.xticks([1, 2], ['Pre-trim', 'Post-trim'])
    plt.title('Spearman correlation coefficiant')
    plt.subplot(1, 2, 2)
    plt.hist(corr_diff, np.arange(-0.5, 0.5, 0.05))
    plt.xlim(-0.5, 0.5)
    plt.title('Change in Spearman correlation coefficiant')

    sp_rej, sp_pval_corr = smm.multipletests(sp_ttest_pvals.reshape(test_pvals.shape[0],), alpha=0.05, method='simes-hochberg')[:2]
    print('set-point fraction significant: ' + str(float(sum(sp_rej))/len(sp_rej)))

    amp_rej, amp_pval_corr = smm.multipletests(amp_ttest_pvals.reshape(test_pvals.shape[0],), alpha=0.05, method='simes-hochberg')[:2]
    print('amplitude fraction significant: ' + str(float(sum(amp_rej))/len(amp_rej)))

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(sp_diff, np.arange(-17, 17), align='left')
    plt.xlim(-15, 15)
    plt.xlabel('set-point difference (deg)')


    plt.subplot(1, 2, 2)
    plt.hist(amp_diff, np.arange(-17, 17, 0.5), align='left')
    plt.xlim(-5, 5)
    plt.xlabel('amplitude difference (deg)')

##### set-point vs runspeed #####
##### set-point vs runspeed #####
    fig = plt.figure(figsize=(24, 14))
    for i in range(1,10):
        plt.subplot(3, 4, i)
        plt.scatter(vel_vals[i-1], set_point_vals[i-1], c='k', alpha=0.7)
        plt.scatter(vel_vals[i-1+9], set_point_vals[i-1+9], c='r', alpha=0.7)
        plt.title('Pos ' + str(i))
        plt.xlim(0, 120)
#        plt.ylim(60, 160)
    plt.show()

    camTime, set_point_list, amp_list, vel_list, ang_list = get_wt_run_vals(wtmat, trials_ran_dict,
            cond_ind_dict, vel_mat, trial_time)

    set_point_cum, amp_cum, angle_cum, run_cum, total_trials = \
            get_cummulative_distributions(wtmat, trials_ran_dict, trial_time,
                    start_time=1.5, stop_time=2.5, control_pos=9)

##### make set-point box-plot       #####
##### From mean set point per trial #####
    sp_vec_list_temp = [np.mean(x[:, 750:1000], axis=1) for x in set_point_list]
    sp_vec_list = [0]*18
    count = 0
    for i in range(18):
        if i%2 == 0:
            sp_vec_list[i] = sp_vec_list_temp[count]
        else:
            sp_vec_list[i] = sp_vec_list_temp[count + 9]
            count += 1

#####  Make tuning curves using mean set-point per trial  #####
#####             make set-point tuning curve             #####

    fig = plt.figure(figsize=(24, 14))

    sp_samples_list  = [np.mean(x[:, 750:1000], axis=1) for x in set_point_list]
    sp_mean_list = [np.nanmean(x) for x in sp_samples_list]
    sp_error_list = [mean_confidence_interval(x)[2] for x in sp_samples_list]

    effect_size  = [cohens_d(sp_samples_list[i], sp_samples_list[i+9])[0] for i in range(9)]
    effect_error = [cohens_d(sp_samples_list[i], sp_samples_list[i+9])[1] for i in range(9)]

    x = np.arange(1, 10)
    # plot tuning curve
    plt.subplot(2,3,1)
    b1 = plt.errorbar(x, sp_mean_list[:9], yerr=sp_error_list[:9], fmt='-o', color='b')
    b2 = plt.errorbar(x+0.25, sp_mean_list[9:], yerr=sp_error_list[9:], fmt='-o', color='r')
    plt.xticks(x)
    plt.xlim(0, 10)
    plt.xlabel('Bar Position')
    plt.ylabel('Set-point +/- 95% CI')
    plt.title('Set-point Tuning curve')
    plt.legend([b1[0], b2[0]], ['Pre-trim', 'Post-trim'], loc='lower right')

    # plot effect size
    plt.subplot(2,3,2)
    plt.bar(x-0.5, effect_size, width=1, yerr=effect_error)
    plt.xticks(x)
    plt.ylim(0, 2)
    plt.xlim(0, 10)
    plt.xlabel('Bar Position')
    plt.ylabel('Cohen\'s D')
    plt.title('Effect Size')

    # plot box-whisker plot
    plt.subplot(2,3,3)
    xx = np.arange(1.5, 18.5, 2)
    xlbls = [str(x) for x in x]
    box = plt.boxplot(sp_vec_list, widths=0.75)
    ylim = plt.ylim()
    plt.vlines(np.arange(2,18,2)+0.5, ymin=ylim[0], ymax=ylim[1])
    plt.xticks(xx, xlbls)
    plt.xlim(0, 19)
    plt.xlabel('Bar Position')
    plt.ylabel('Set-point (deg)')
    plt.title('Set-point Box and Whisker Plot')

##### make amplitude box-plot        #####
##### From mean amplitude  per trial #####
    amp_vec_list_temp = [np.mean(x[:, 750:1000], axis=1) for x in amp_list]
    amp_vec_list = [0]*18
    count = 0
    for i in range(18):
        if i%2 == 0:
            amp_vec_list[i] = amp_vec_list_temp[count]
        else:
            amp_vec_list[i] = amp_vec_list_temp[count + 9]
            count += 1

#####  Make tuning curves using mean amplitude per trial  #####
#####             make amplitude tuning curve             #####

    amp_samples_list  = [np.nanmean(x[:, 750:1000], axis=1) for x in amp_list]
    amp_mean_list = [np.nanmean(x) for x in amp_samples_list]
    amp_error_list = [mean_confidence_interval(x)[2] for x in amp_samples_list]

    effect_size  = [cohens_d(amp_samples_list[i], amp_samples_list[i+9])[0] for i in range(9)]
    effect_error = [cohens_d(amp_samples_list[i], amp_samples_list[i+9])[1] for i in range(9)]

    x = np.arange(1, 10)
    # plot tuning curve
    plt.subplot(2,3,4)
    b1 = plt.errorbar(x, amp_mean_list[:9], yerr=amp_error_list[:9], fmt='-o', color='b')
    b2 = plt.errorbar(x+0.25, amp_mean_list[9:], yerr=amp_error_list[9:], fmt='-o', color='r')
    plt.xticks(x)
    plt.xlim(0, 10)
    plt.xlabel('Bar Position')
    plt.ylabel('Amplitude +/- 95% CI')
    plt.title('Amplitude Tuning curve')
    plt.legend([b1[0], b2[0]], ['Pre-trim', 'Post-trim'], loc='lower right')

    # plot effect size
    plt.subplot(2,3,5)
    plt.bar(x-0.5, effect_size, width=1, yerr=effect_error)
    plt.xticks(x)
    plt.ylim(0, 2)
    plt.xlim(0, 10)
    plt.xlabel('Bar Position')
    plt.ylabel('Cohen\'s D')
    plt.title('Effect Size')

    # plot box-whisker plot
    plt.subplot(2,3,6)
    xx = np.arange(1.5, 18.5, 2)
    xlbls = [str(x) for x in x]
    box = plt.boxplot(amp_vec_list, widths=0.65)
    ylim = plt.ylim()
    plt.vlines(np.arange(2,18,2)+0.5, ymin=ylim[0], ymax=ylim[1])
    plt.xticks(xx, xlbls)
    plt.xlim(0, 19)
    plt.ylabel('Amplitude (deg)')
    plt.title('Amplitude Box and Whisker Plot')
    plt.suptitle(fid)
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.93, wspace=0.17, hspace=0.25)
    plt.show()


#####  For all cummulative data points during stim period #####
#####             make set-point tuning curve             #####

    x   = np.arange(1, 10)
    y   = np.asarray([np.nanmean(y) for y in set_point_cum])
    err = np.asarray([np.nanstd(err) for err in set_point_cum])

    ydiff   = y[9:] - y[:9]
    errdiff = np.sqrt(err[:9]**2 + err[9:]**2)

    plt.figure()
    plt.errorbar(x, y[:9], yerr=err[:9], fmt='-o', color='b')
    plt.errorbar(x+0.25, y[9:], yerr=err[9:], fmt='-o', color='r')
    plt.xlim(0, 10)
    plt.title('Set-point Tuning curve\nof cummulative data (+/- sigma)')
    plt.show()

    plt.figure()
    plt.errorbar(x, ydiff, yerr=errdiff, fmt='-o', color='k')
    plt.hlines(0, 0, 10, color='k')
    plt.xlim(0, 10)
    plt.title('Difference tuning curve\ncummulative set-point data (+/- sigma)')
    plt.show()

#################### set-point diifference tuning curve  ######################
#################### run speed difference tuning curve   ######################

    fig = plt.figure(figsize=(24, 14))
    # set-point difference
    plt.subplot(1,2,1)
    sp_samples_list  = [np.mean(x[:, 750:1000], axis=1) for x in set_point_list]
    sp_mean_list = [np.nanmean(x) for x in sp_samples_list]
    sp_mean_diff = [sp_mean_list[i] - sp_mean_list[i+9] for i in range(9)]
    sp_error_diff = list()
    for i in range(9):
        n1 = len(sp_samples_list[i])
        n2 = len(sp_samples_list[i+9])
        var1 = np.nanvar(sp_samples_list[i])
        var2 = np.nanvar(sp_samples_list[i+9])
        var_pool = ( (n1 - 1)*var1 + (n2 - 1)*var2)/(n1 + n2 -2)
        var_err  = np.sqrt(var_pool/n1 + var_pool/n2)
        sp_error_diff.append(var_err*sp.stats.t.ppf((1+0.95)/2.0, n1+n2-2))
    x = np.arange(1, 10)
    # plot tuning curve
    b1 = plt.errorbar(x, sp_mean_diff[:9], yerr=sp_error_diff[:9], fmt='-o', color='k')
    plt.xticks(x)
    plt.xlim(0, 10)
    plt.hlines(0, 0, 10, linestyles='dashed')
    plt.xlabel('Bar Position')
    plt.ylabel('Set-point Difference +/- 95% CI')
    plt.title('Set-point Difference Tuning curve')


    plt.subplot(1,2,2)
    vl_samples_list  = [np.mean(x[:, 750:1000], axis=1) for x in vel_list]
    vl_mean_list = [np.nanmean(x) for x in vl_samples_list]
    vl_mean_diff = [vl_mean_list[i] - vl_mean_list[i+9] for i in range(9)]
    vl_error_diff = list()
    for i in range(9):
        n1 = len(vl_samples_list[i])
        n2 = len(vl_samples_list[i+9])
        var1 = np.nanvar(sp_samples_list[i])
        var2 = np.nanvar(sp_samples_list[i+9])
        var_pool = ( (n1 - 1)*var1 + (n2 - 1)*var2)/(n1 + n2 -2)
        var_err  = np.sqrt(var_pool/n1 + var_pool/n2)
        vl_error_diff.append(var_err*sp.stats.t.ppf((1+0.95)/2.0, n1+n2-2))
    x = np.arange(1, 10)
    # plot tuning curve
    b1 = plt.errorbar(x, vl_mean_diff[:9], yerr=vl_error_diff[:9], fmt='-o', color='k')
    plt.xticks(x)
    plt.xlim(0, 10)
    plt.hlines(0, 0, 10, linestyles='dashed')
    plt.xlabel('Bar Position')
    plt.ylabel('Velocity Difference +/- 95% CI')
    plt.title('Velocity Difference Tuning curve')
    plt.show()

##### make bootstrap distributions of the differences in medians

#    plt.figure()
#    for pos in range(9):
#        boot_dist = bootstrap_medians(set_point_cum[pos], set_point_cum[pos + 9])
#        median_diff = np.abs(np.median(set_point_cum[pos]) - np.median(set_point_cum[pos + 9]))
#        p_val = (sum(boot_dist > median_diff) + sum(boot_dist < -median_diff))/len(boot_dist)
#        plt.subplot(1, 9, pos+1)
#        plt.hist(boot_dist, 100)
#        plt.vlines(median_diff, 0, plt.ylim()[1], linestyles='dashed', colors='r')
#        plt.xticks(rotation=45)
#        plt.title('p < ' +  "{:.4f}".format(p_val))
#
#    plt.suptitle('Bootstrap distributions of differences in medians')
#    plt.show()

########################## set-point dist overlay #############################
########################## run speed dist overlay #############################

    binsize = 2
    bin_start = 90
    bin_stop  = 160
    runbinsize = 50
    run_bin_start = 0
    run_bin_stop  = 1250
    bins = np.arange(bin_start, bin_stop, binsize)
    run_bins = np.arange(run_bin_start,run_bin_stop, runbinsize)
    plt.figure()

    xset = np.arange(90, 160, 20)
    xsetlbls = [str(x) for x in xset]
    xrun = np.arange(0, 1100, 200)
    xrunlbls = [str(x) for x in xrun]

    for pos in range(1,10):
        set_point_cum_hist = np.histogram(set_point_cum[pos-1], bins)[0]*0.002
        set_point_cum_hist_trim = np.histogram(set_point_cum[pos-1+9], bins)[0]*0.002
        set_point_max = np.nanmax(np.append(set_point_cum_hist/total_trials[pos-1],set_point_cum_hist_trim/total_trials[pos-1+9]))
        plt.subplot(2,9,pos)
        plt.plot(bins[:-1], set_point_cum_hist/total_trials[pos-1], 'b');plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max)

        if pos == 1:
            plt.ylabel('seconds/(bin*trial)')
            plt.title('set-point (deg)')

        plt.plot(bins[:-1], set_point_cum_hist_trim/total_trials[pos-1+9],'r');plt.xlim(bin_start, bin_stop)
        plt.ylim(0, set_point_max + set_point_max*0.1)
        plt.xticks(xset, xsetlbls)

        run_pre_trim_hist = np.histogram(run_cum[pos-1], run_bins)[0]/total_trials[pos-1]
        run_post_trim_hist = np.histogram(run_cum[pos-1+9], run_bins)[0]/total_trials[pos-1+9]
        run_max_ylim = np.nanmax(np.append(run_pre_trim_hist, run_post_trim_hist))
        plt.subplot(2,9,pos+9)
        plt.plot(run_bins[:-1], run_pre_trim_hist,'b');plt.xlim(run_bin_start, run_bin_stop)
        plt.plot(run_bins[:-1], run_post_trim_hist,'r');plt.xlim(run_bin_start, run_bin_stop)
        plt.ylim(0, run_max_ylim + run_max_ylim*0.1)

        if pos == 1:
            plt.ylabel('counts/(bin*trial)')
            plt.title('runspeed (deg/sec)')

        plt.xticks(xrun, xrunlbls, rotation=45)

    plt.suptitle(fid + ' cumulative set-point and runspeed distributions')
    plt.show()

##### test PSD confidence interval stuff  #####
    params.Fs = 500;
    params.tapers = [3 5];
    params.fpass = [1:250];
    params.err = [1 0.95];
    params.trialave = 1;

    test = ang_dict['1062']
    pos = test[0].T





##### test PSD confidence interval stuff  #####
def angle_psd(ang_list, camTime, start=1.5, stop=2.5):
    analysis_inds = np.where(np.logical_and(camTime >= start, camTime <= stop))
    pxx_list = list()
    for angle_traces in ang_list:
        Pxx_all = np.zeros((257, 0))
        for i, trace in enumerate(angle_traces):
            if np.sum(np.where(np.isnan(trace))[0]) == 0:
                padded_trace = np.pad(trace, ((100,100)), mode='constant', constant_values=0)
                Fxx, Pxx = sp.signal.welch(padded_trace, fs=500, window='hanning', nperseg=512,
                        noverlap=256, detrend='constant')
                Pxx_all = np.concatenate((Pxx_all, Pxx.reshape(257,1)), axis=1)
        pxx_list.append(Pxx_all)

    pxx = list()
    for trial, pxx_all in enumerate(pxx_list):
        print('on trial ' + str(trial))
        pxxc_upper = list()
        pxxc_lower = list()
        boot_samples = np.zeros((10000, 257))
        n = pxx_all.shape[1]
        for sample in range(10000):
            for i in range(257):
                boot_samples[sample, i] = np.mean(np.random.choice(pxx_all[i,:], n))

        # Calculate 95% Confidence Interval from bootstrap distribution
        for i in range(257):
            temp = np.percentile(boot_samples[:, i], [0.025, 0.975])
            pxxc_upper.append(temp[1])
            pxxc_lower.append(temp[0])
        pxx.append( (pxx_all.mean(axis=1), pxxc_lower, pxxc_upper))


#    Pxx_all_mean = Pxx_all.mean(axis=1)
#    P = 257 # number of estimates in the Welch function also degree of freedom
#    alpha = 1 - 0.95
#    v = 2*P
#    c = sp.stats.chi2.ppf([1 - alpha / 2.0, alpha / 2.0], v)
#    c = v / c
#    Pxxc_lower = Pxx_all_mean*c[0]
#    Pxxc_upper = Pxx_all_mean*c[1]
    plt.figure()
    for t in range(9):
        plt.subplot(3,4,t+1)
        plt.plot(Fxx, pxx[t][0], 'k')
        plt.fill_between(Fxx, pxx[t][0]- pxx[t][1], pxx[t][0]+ pxx[t][2],
                facecolor='k', alpha=0.3)
        plt.plot(Fxx, pxx[t+9][0], 'r')
        plt.fill_between(Fxx, pxx[t+9][0]- pxx[t+9][1], pxx[t+9][0]+ pxx[t+9][2],
                facecolor='r', alpha=0.3)
        plt.xlim(0,35)
        plt.yscale('log')
    plt.show()


##### set-point spike triggered averages  #####
#    pos = 9
#    binsize = 2
#    bin_start = 120
#    bin_stop  = 160
#
#    for unit in range(len(set_point_sta)):
#        bins = np.arange(bin_start, bin_stop, binsize)
#        set_point_sta_hist = np.histogram(set_point_sta[unit][pos-1], bins)[0]
#        set_point_cum_hist = np.histogram(set_point_cum[pos-1], bins)[0]*0.002
#        set_point_norm = set_point_sta_hist/set_point_cum_hist
#
#        set_point_sta_hist_trim = np.histogram(set_point_sta[unit][pos-1+9], bins)[0]
#        set_point_cum_hist_trim = np.histogram(set_point_cum[pos-1+9], bins)[0]*0.002
#        set_point_norm_trim = set_point_sta_hist_trim/set_point_cum_hist_trim
#
#        sta_max = np.nanmax(np.append(set_point_sta_hist, set_point_sta_hist_trim))
#        cum_max = np.nanmax(np.append(set_point_cum_hist, set_point_cum_hist_trim))
#        nor_max = np.nanmax(np.append(set_point_norm, set_point_norm_trim))
#
#        plt.figure()
#
#        plt.subplot(3,2,1)
#        plt.bar(bins[:-1], set_point_sta_hist, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, sta_max)
#        plt.title('Pre-trim')
#        plt.ylabel('spikes/bin')
#        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#        plt.subplot(3,2,3)
#        plt.bar(bins[:-1], set_point_cum_hist, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, cum_max)
#        plt.ylabel('seconds/bin')
#        plt.subplot(3,2,5)
#        plt.bar(bins[:-1], set_point_norm, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, nor_max)
#        plt.tick_params(axis='x', which='both', top='off')
#        plt.ylabel('spike rate (Hz)')
#        plt.xlabel('set-point (deg)')
#
#        plt.subplot(3,2,2)
#        plt.bar(bins[:-1], set_point_sta_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, sta_max)
#        plt.title('Post-trim')
#        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#        plt.subplot(3,2,4)
#        plt.bar(bins[:-1], set_point_cum_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, cum_max)
#        plt.subplot(3,2,6)
#        plt.bar(bins[:-1], set_point_norm_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, nor_max)
#        plt.tick_params(axis='x', which='both', top='off')
#        plt.xlabel('set-point (deg)')
#
#        plt.show()

##### amplitude spike triggered averages  #####
#    pos = 9
#    binsize = 1
#    bin_start = 0
#    bin_stop  = 25
#
#    for unit in range(len(amp_sta)):
#        bins = np.arange(bin_start, bin_stop, binsize)
#        amp_sta_hist = np.histogram(amp_sta[unit][pos-1], bins)[0]
#        amp_cum_hist = np.histogram(amp_cum[pos-1], bins)[0]*0.002
#        amp_norm = amp_sta_hist/amp_cum_hist
#
#        amp_sta_hist_trim = np.histogram(amp_sta[unit][pos-1+9], bins)[0]
#        amp_cum_hist_trim = np.histogram(amp_cum[pos-1+9], bins)[0]*0.002
#        amp_norm_trim = amp_sta_hist_trim/amp_cum_hist_trim
#
#        sta_max = np.nanmax(np.append(amp_sta_hist, amp_sta_hist_trim))
#        cum_max = np.nanmax(np.append(amp_cum_hist, amp_cum_hist_trim))
#        nor_max = np.nanmax(np.append(amp_norm, amp_norm_trim))
#
#        plt.figure()
#
#        plt.subplot(3,2,1)
#        plt.bar(bins[:-1], amp_sta_hist, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, sta_max)
#        plt.title('Pre-trim')
#        plt.ylabel('spikes/bin')
#        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#        plt.subplot(3,2,3)
#        plt.bar(bins[:-1], amp_cum_hist, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, cum_max)
#        plt.ylabel('seconds/bin')
#        plt.subplot(3,2,5)
#        plt.bar(bins[:-1], amp_norm, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, nor_max)
#        plt.tick_params(axis='x', which='both', top='off')
#        plt.ylabel('spike rate (Hz)')
#        plt.xlabel('amp (deg)')
#
#        plt.subplot(3,2,2)
#        plt.bar(bins[:-1], amp_sta_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, sta_max)
#        plt.title('Post-trim')
#        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#        plt.subplot(3,2,4)
#        plt.bar(bins[:-1], amp_cum_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, cum_max)
#        plt.subplot(3,2,6)
#        plt.bar(bins[:-1], amp_norm_trim, binsize);plt.xlim(bin_start, bin_stop)
#        plt.ylim(0, nor_max)
#        plt.tick_params(axis='x', which='both', top='off')
#        plt.xlabel('amp (deg)')
#
#        plt.show()

    fids = ['1034', '1044', '1054', '1058', '1062']
    region = 'vS1'

    sns.set_context("poster")
    scorr_coefs = np.empty((0, 2))
    test_pvals  = np.empty((0, 1))

    for fid in fids:
        usr_dir = os.path.expanduser('~')
        sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
        fid_region = 'fid' + fid + '_' + region
        sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

        data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
        data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

        # Calculate runspeed
        run_mat = load_run_file(data_dir_paths[0]).value
        vel_mat, trial_time = calculate_runspeed(run_mat)

        # Get stimulus id list
        stim = load_stimsequence(data_dir_paths[0])

        # Create running trial dictionary
        # Strict running thresholds
        cond_ind_dict,trials_ran_dict = classify_run_trials(stim, vel_mat,
                trial_time, stim_start=1.25, stim_stop=2.50, mean_thresh=400,
                sigma_thresh=150, low_thresh=200, display=False)

        df = make_df(sort_file_paths,data_dir_path,region=region)
        vel_mat = vel_mat*(2*np.pi*6)/360.0

        hsv_mat_path = glob.glob(usr_dir + '/Documents/AdesnikLab/Processed_HSV/FID' + fid + '-data*.mat')[0]
        wtmat = h5py.File(hsv_mat_path)
        camTime, set_point_list, amp_list, vel_list, ang_list = get_wt_run_vals(wtmat, trials_ran_dict,
                cond_ind_dict, vel_mat, trial_time)
        sp_sta, amp_sta, ang_sta = get_spike_triggered_averages(df, wtmat, trials_ran_dict,
                trial_time, start_time=1.5, stop_time=2.5, control_pos=9)
        set_point_cum, amp_cum, ang_cum, run_cum, total_trials = \
                get_cummulative_distributions(wtmat, trials_ran_dict, trial_time,
                        start_time=1.5, stop_time=2.5, control_pos=9)


##### angle spike triggered averages  #####
    pos = 9
    binsize = 5
    bin_start = 70
    bin_stop  = 180
    bins = np.arange(bin_start, bin_stop, binsize)

    for unit in range(len(ang_sta)):
        plt.figure()
        for i,pos in enumerate(range(9)):
            ang_sta_hist = np.histogram(ang_sta[unit][pos-1], bins)[0]
            ang_cum_hist = np.histogram(ang_cum[pos-1], bins)[0]*0.002
            ang_norm = ang_sta_hist/ang_cum_hist

            ang_sta_hist_trim = np.histogram(ang_sta[unit][pos-1+9], bins)[0]
            ang_cum_hist_trim = np.histogram(ang_cum[pos-1+9], bins)[0]*0.002
            ang_norm_trim = ang_sta_hist_trim/ang_cum_hist_trim

            plt.subplot(2,5,i+1)
            plt.plot(bins[:-1], ang_norm, 'k', bins[:-1], ang_norm_trim, 'r')
            plt.xlim(90, 165)

        plt.show()

#            sta_max = np.nanmax(np.append(ang_sta_hist, ang_sta_hist_trim))
#            cum_max = np.nanmax(np.append(ang_cum_hist, ang_cum_hist_trim))
#            nor_max = np.nanmax(np.append(ang_norm, ang_norm_trim))
#
#            plt.figure()
#
#            plt.subplot(3,2,1)
#            plt.bar(bins[:-1], ang_sta_hist, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, sta_max)
#            plt.title('Pre-trim')
#            plt.ylabel('spikes/bin')
#            plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#            plt.subplot(3,2,3)
#            plt.bar(bins[:-1], ang_cum_hist, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, cum_max)
#            plt.ylabel('seconds/bin')
#            plt.subplot(3,2,5)
#            plt.bar(bins[:-1], ang_norm, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, nor_max)
#            plt.tick_params(axis='x', which='both', top='off')
#            plt.ylabel('spike rate (Hz)')
#            plt.xlabel('ang (deg)')
#
#            plt.subplot(3,2,2)
#            plt.bar(bins[:-1], ang_sta_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, sta_max)
#            plt.title('Post-trim')
#            plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
#            plt.subplot(3,2,4)
#            plt.bar(bins[:-1], ang_cum_hist_trim, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, cum_max)
#            plt.subplot(3,2,6)
#            plt.bar(bins[:-1], ang_norm_trim, binsize);plt.xlim(bin_start, bin_stop)
#            plt.ylim(0, nor_max)
#            plt.tick_params(axis='x', which='both', top='off')
#            plt.xlabel('ang (deg)')
#
#            plt.show()


















