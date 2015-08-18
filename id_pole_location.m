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
%% Specify video directory and select .dat file
%HAVE USER SPECIFY WHICH ID IS THE CONTROL POSITION
prompt    = {'First Control Position'};
dlgTitle  = 'Control Position';
numLines  = 1;
default   = {'9'};
usrInput  = inputdlg(prompt,dlgTitle,numLines,default);

if isempty(usrInput)
    disp('process canceled')
    return
end

control_pos = str2double(usrInput{1});

% Have user specify the video directory
% UPDATE THIS TO THE USUAL VIDEO DIRECTORY
vidDir = uigetdir('E:\','Select the video directory to analyze');

[~,fileName] = fileparts(vidDir);

% List Directories Contents
vidStrct = dir([vidDir filesep '*.seq']);

if vidDir == 0
    error('video directory not selected')
elseif isempty(vidStrct)
    error('video directory is empty!')
end

[dat_filename, dat_path] = uigetfile('.dat', 'Select .dat file for this experiment');
load([dat_path dat_filename], '-mat', 'stimsequence');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make whisker density matrices for each trial type

num_stimuli = length(unique(stimsequence));
pos_centers = nan(control_pos, 1);

for pos = 1:control_pos
    stim_ind = find(stimsequence == pos);
    % Load in one movie frame to get image size for array pre-allocation
    vid_fname = vidStrct(stim_ind).name;
    [~, img] = Norpix2MATLABopenSingleFrame([vidDir filesep vid_fname], 1);
    h1 = figure('position', [0, 0, dim(2), dim(1)]);
    imagesc(imgOut);
    title({['Click on location on pole where the whisker makes contact for position ' num2str(pos)],
    'Click on upper left corner if pole is not visible'})
    pos_centers(pos) = ginput(1);
    close all;
end

save([dat_filename(1:end-4) 'pole_positions.mat'], 'pos_centers', '-mat');







