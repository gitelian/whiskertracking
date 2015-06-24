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
analStopInd  = floor((analStop - startTime)/timeStep) + 1;

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
trial_counter = 0
num_good_trials = length(trialsran);

for trial_type = 1:num_stimuli

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
        threshold = mean(temp) + 4*std(single(temp));
        clear temp

        % Pre-allocate 'counts' matrix on first movie
        if k == 1
            counts = zeros(size(img,1), size(img,2));
            frame_counter = 0;
        end

        % Compute whisker counts for density calculation
        % Only during specified analysis period
        for frind = analStartInd:analStopInd
            temp = img(:,:,frind);
            img_inds = temp > threshold;
            counts(img_inds) = counts(img_inds) + 1;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dat_filename, dat_path] = uigetfile('.den', 'Select whisker density file');
load([dat_path dat_filename], '-mat');

num_stimuli   = size(whisker_density, 3);
set_point     = zeros(length(num_stimuli), 1);
amp           = zeros(length(num_stimuli), 1);
sector_coords = cell(num_stimuli);

for k = 1:num_stimuli
    figure('position', [0, 0, size(whisker_density,2), size(whisker_density,1)])
    imagesc(1-whisker_density(:,:,k), [0.8 1]);
    colormap gray
    title({['Stimulus: ' num2str(k) ' ' 'Define the edges of the sector'],...
        '!!! The first point of an edge should be closest to the whisker pad !!!'})

    [x, y] = ginput(4);
    sector_coords{k} = [x, y];
    close all
    vec1 = [x(2) - x(1), -(y(2) - y(1))];
    vec2 = [x(4) - x(3), -(y(4) - y(3))];

    cos_theta = dot(vec1, vec2)/(norm(vec1)*norm(vec2));
    amp_temp = acos(cos_theta)*180/pi;
    amp(k) = amp_temp;
    % Evan YOU HAVE TO CHANGE THIS LINE TO CALCULATE THE CORRECT SET-POINT FOR
    % YOU SETUP
    max_angle = max([acos(dot(vec1,[1 0])/norm(vec1))*180/pi,...
                    acos(dot(vec2, [1 0])/norm(vec2))*180/pi]);
    set_point(k) = max_angle - amp_temp/2;

end

save([dat_filename(1:end-4) '-density-set-point.mat'], '-v7.3')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic()
%white_background = 0;
%
%[~, img] = Norpix2MATLAB([vid_dir filesep vid_fname]);
%img = uint8(img);
%temp = reshape(img, size(img,1)*size(img,2)*size(img,3), 1);
%
%if white_background
%    threshold = median(temp)*3/4;
%else
%    threshold = mean(temp) + 4*std(single(temp));
%end
%
%clear temp
%
%counts = zeros(size(img,1), size(img,2));
%
%for frind = 1:size(img,3)
%    temp = img(:,:,frind);
%    if white_background
%        img_inds = temp < threshold;
%    else
%        img_inds = temp > threshold;
%    end
%    counts(img_inds) = counts(img_inds) + 1;
%end
%
%mean_counts = counts/size(img,3);
%
%% For regular colors
%figure
%imagesc(mean_counts, [0,0.1])
%colormap hot
%
%% For inverted colors
%figure
%imagesc(abs(1 - mean_counts), [0.9,1])
%colormap gray
%
%% Use Hough transform to get whisker trace
%angles = nan(size(img,3),1);
%lines  = cell(size(img,3),1);
%parfor f = 1:size(img, 3)
%    temp = img(:,:,f);
%    temp = im2bw(temp, 0.2);
%    temp = imfill(temp, 'holes');
%    BW = edge(temp, 'canny');
%    [H, theta, rho] = hough(BW);
%    P = houghpeaks(H,1,'threshold', ceil(0.3*max(H(:))));
%    l = houghlines(BW, theta, rho, P, 'FillGap', 150, 'MinLength', 100);
%    if ~isempty(l)
%        xy1 = l.point1;
%        xy2 = l.point2;
%        if xy1(2) > xy2(2)
%            xy_new = xy1 - xy2;
%        elseif xy1(2) < xy2(2)
%            xy_new = xy2 - xy1;
%        end
%        ang = 180 - atan2(xy_new(2), xy_new(1))*180/pi;
%        angles(f,1) = ang;
%        lines{f,1} = l;
%    end
%end
%toc()
%t = 0:size(img, 3) - 1;
%figure;
%plot(t, angles);
%ylim([90 180])
%
%% Code that attempts to find the sector boundaries of a 2d histogram whisker
%% trace. Play around with this IF automated computing of sector boundaries is
%% important (i.e. if we want to calculate the mean set-point across all trials.
%% Maybe there is an easier way: draw a small rectangular roi (or line) and
%% calculate the region boundary using three points. The first point is the max
%% protraction point, second point is the min protraction point, and the third
%% point is a user define point placed on base of the whisker follicle. Using
%% these three points I can calculate the sector angle (amplitude), and
%% set-point.
%%
%%BW1 = im2bw(mean_counts, 1/6*(threshold/255));
%%BW = imfill(BW1, 'holes');
%%figure;imagesc(BW);colormap gray
%%BW = edge(BW, 'canny');
%%[H,theta,rho] = hough(BW);
%%P = houghpeaks(H,2,'threshold',ceil(0.3*max(H(:))));
%%lines = houghlines(BW,theta,rho,P,'FillGap',150,'MinLength',100);
%%figure;colormap gray
%%imagesc(abs(1 - mean_counts), [0.9,1])
%%hold on
%%max_len = 0;
%%for k = 1:length(lines)
%%    xy = [lines(k).point1; lines(k).point2];
%%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%%
%%    % Plot beginnings and ends of lines
%%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
%%
%%    % Determine the endpoints of the longest line segment
%%    len = norm(lines(k).point1 - lines(k).point2);
%%    if ( len > max_len)
%%        max_len = len;
%%        xy_long = xy;
%%    end
%%end
%%hold off
%%toc()
