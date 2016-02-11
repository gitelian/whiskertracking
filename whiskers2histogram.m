%WHISKERS2HISTOGRAM Convert high speed videos of whiskers to 2d histograms.
%   Have the user enter details of the experiment, and specify where the video
%   files are located, and which .dat file corresponds to those videos. This
%   script will then measure the whisker density for running trials during the
%   specified analysis period.
%
%   Saves a 3-d matrix in the local directory. The third dimension corresponds
%   to each trial type. To visualize the density map you can use imagesc like
%   so: figure; imagesc(whisker_density(:,:,1), [0 0.2]), colormap hot
%
%   Dependancies: Norpix2MATLABopenSingleFrame, Norpix2MATLAB,
%   classify_run_trials, MakeGaussWindow, fwhm.
%
%   UC Berkeley
%   Adesnik Lab
%   G. Telian
%   20150624

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Have the user verify/change video paramenters
%  Find the start and stop indices from the specified analysis period. Only
%  frames from this period will be used in constructing the density plot.

prompt    = {'video start time', 'video stop time', 'analysis start time',...
    'analysis stop time', 'fps',  'e-phys sampling rate', 'runspeed threshold'};
dlgTitle  = 'High speed video parameters';
numLines  = 1;
default   = {'0.5', '2.5', '1.5', '2.5', '500', '30000', '250'};
usrInput  = inputdlg(prompt,dlgTitle,numLines,default);

if isempty(usrInput)
    disp('process canceled')
    return
end

startTime = str2double(usrInput{1});
stopTime  = str2double(usrInput{2});
analStart = str2double(usrInput{3});
analStop  = str2double(usrInput{4});
fps       = str2double(usrInput{5});
sr        = str2double(usrInput{6});
runthresh = str2double(usrInput{7});

timeStep  = 1/fps; % may need to add e-phys sampling rate?round(1/fps*sr)/sr

camTime      = startTime:timeStep:(stopTime-timeStep);
analStartInd = floor((analStart - startTime)/timeStep) +1; % +1 to deal with zero not working as a first frame index
analStopInd  = floor((analStop - startTime)/timeStep);% + 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Specify video directory and select .dat file

% Have user specify the video directory
% UPDATE THIS TO THE USUAL VIDEO DIRECTORY
vidDir = uigetdir('E:\','Select the video directory to analyze');

%% REMOVE THIS %%
%vidDir = '/media/greg/Data/HSV_Videos/FID0964'

[~,fileName] = fileparts(vidDir);

% List Directories Contents
vidStrct = dir([vidDir filesep '*.seq']);

if vidDir == 0
    error('video directory not selected')
elseif isempty(vidStrct)
    error('video directory is empty!')
end

[dat_filename, dat_path] = uigetfile('.dat', 'Select .dat file for this experiment');
load([dat_path dat_filename], '-mat', 'stimsequence', 'run_data');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make whisker density matrices for each trial type

% Get running trials
[trialsran, ~, ~, ~] = classify_run_trials(run_data, sr, runthresh, 25,...
    analStart, analStop, 0);

num_stimuli = length(unique(stimsequence));

% Load in one movie frame to get image size for array pre-allocation
vid_fname = vidStrct(1).name;
[~, img] = Norpix2MATLABopenSingleFrame([vidDir filesep vid_fname], 1);
whisker_density = zeros(size(img, 1), size(img, 2), num_stimuli);

h = waitbar(0, 'Initializing waitbar ...');
trial_counter = 0;
num_good_trials = length(trialsran);

for trial_type = 6;%1:num_stimuli

    good_trial_indices = intersect(find(stimsequence == trial_type), trialsran);

    % Iterate through all movies of a particular trial type
    for k = 1:length(good_trial_indices)

        % load movie
        vid_fname = vidStrct(good_trial_indices(k)).name;
        [~, img] = Norpix2MATLAB([vidDir filesep vid_fname]);
        img = uint8(img);

        % Compute threshold
        % used to binarize image and separate reflective whisker from
        % background and noise.
        temp = reshape(img, size(img,1)*size(img,2)*size(img,3), 1);
        threshold = mean(temp) + 1.5*std(single(temp));
        %threshold = 5;
        clear temp

        % Pre-allocate 'counts' matrix on first movie
        if k == 1
            counts = zeros(size(img,1), size(img,2));
            frame_counter = 0;
        end

        % Compute whisker counts for density calculation
        % Only during specified analysis period

%         for frind = analStartInd:analStopInd
        for frind = 1:737
            temp = img(:,:,frind);
            img_inds = temp > threshold;
%             counts(img_inds) = counts(img_inds) + 1;
            counts = counts + double(img(:,:,frind));

            frame_counter = frame_counter + 1;
        end
        % End Compute whisker counts for density calculation

        % Compute mean density from counts
        if k == length(good_trial_indices)
            whisker_density(:, :, trial_type) = counts/(frame_counter);
        end

        % Update the progress bar
        trial_counter = trial_counter + 1;
        waitbar(trial_counter/num_good_trials, h, sprintf('%d of %d trials processed',...
            [trial_counter num_good_trials]));

    end
end

% Close the progress bar
close(h)

save([dat_filename(1:end-4) '.den'], 'whisker_density', '-v7.3');







