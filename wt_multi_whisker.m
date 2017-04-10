%%WT_MULTI_WHISKER Tracks pad angle from full pad high speed imaging
% run script to calculate smoothed angle, set-point, amplitude, phase, and
% velocity of whisker pad. This script expects tracking_data and .avi files
% for a given experiment.
%
% This will only work for an experiment with NO DROPPED FRAMES!!!
%
% UPDATE 20170410: Each trial should have a single video. Data from each
% trial will be processed independently and stored in a cell array specific
% each measurement (e.g. angle, set-point, phase).
%
% G. Telian

% Adesnik Lab
% UC Berkeley
% 20160921



%% get paths to measure, trace, and avi directories
% prctle = 60;
prctle = 20; % for one whisker

exp_dir = uigetdir('F:\tracking_data', 'select experiment directory');
[~, exp_name, ~] = fileparts(exp_dir);

% measure path
msr_dir = dir([exp_dir filesep 'measure' filesep '*.measurements']);
if isempty(msr_dir)
    error('Measurement directory empty!!!')
end

% trace path
trc_dir = dir([exp_dir filesep 'trace' filesep '*.whiskers']);
if isempty(trc_dir)
    error('Trace directory empty!!!')
end

% xml_path
xml_dir = [exp_dir filesep 'dropped_frames' filesep];
if ~exist(xml_dir, 'dir')
    warning('no dropped frame directory found')
    dropped_frames = 0;
else
    dropped_frames = 1;
    warning('dropped frames found!!!')
end

% avi path
avi_dir = dir(['F:\wtavi' filesep exp_name filesep '*.avi']);
if isempty(avi_dir)
    error('avi directory empty!!!')
end

fname = msr_dir(1).name(1:7);

%% prompt user to select a point on the face of the mouse
fprintf('WAIT!!!\nTHIS MAY TAKE A WHILE TO LOAD\nYOU NEED TO PROVIDE A ROI\n')
msr = struct2table(LoadMeasurements([exp_dir filesep 'measure' filesep msr_dir(1).name]));
trc = struct2table(LoadWhiskers([exp_dir filesep 'trace' filesep trc_dir(1).name]));

v = VideoReader(['F:\wtavi' filesep exp_name filesep avi_dir(1).name]);
fig = figure('Position', [100, 100, v.Width*2, v.Height*2]);
ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;
k = v.NumberOfFrames;
vidFrame = read(v, k);
image(vidFrame);
currAxes.Visible = 'off';
hold on

% get indices for data corresponding to frame k
data_inds = find(msr.fid == k-1);
% plot all whisker traces
for l = data_inds(1):data_inds(end)
    plot(trc.x{l}, trc.y{l}, 'b')
    title(num2str(l))
end

% get length of whisker traces for frame k
% sort whisker traces by longest to shortes
wids = msr.wid(data_inds);
len  = msr.length(data_inds);
ind  = find(len >= prctile(len, prctle));

% get follicle coordinates
x = msr.follicle_x(data_inds(ind));
y = msr.follicle_y(data_inds(ind));
plot(x,y,'co')

for l = 1:length(ind)
    wid = wids(ind(l));
    trc_ind = find(table2array(trc(data_inds, 'id')) == wid);
    plot(trc.x{data_inds(trc_ind)}, trc.y{data_inds(trc_ind)},'--r')
end

title('select pad, then nose, then back of pad')
[x0, y0] = getpts(fig);
xnose = x0(2);
xrear_pad = x0(3);

x0 = x0(1);
y0 = y0(1);
face = [x0, y0];
close all
clear v trc msr
% x0 = 1375;
% y0 = 550;


%% measure angle and calculate smooth angle trace
disp('smoothing angle trace')
angle_notsmooth = cell(length(msr_dir));
ang = cell(length(msr_dir));
wsk = cell(length(msr_dir));
phs = cell(length(msr_dir));
sp = cell(length(msr_dir));
amp = cell(length(msr_dir));
vel = cell(length(msr_dir));

progressbar('files', 'frames')
for f_ind = 1:length(msr_dir)
    progressbar(f_ind/length(msr_dir), [])
    msr = struct2table(LoadMeasurements([exp_dir filesep 'measure' filesep msr_dir(f_ind).name]));
    num_frames = length(unique(msr.fid));
    ang_temp = zeros(num_frames, 1);
    for k = 1:num_frames %v.NumberOfFrames
        progressbar([], k/num_frames)
        % get indices for data corresponding to frame k
        data_inds = find(msr.fid == k-1);
        % get length of whisker traces for frame k
        wids = msr.wid(data_inds);
        len  = msr.length(data_inds);
        ind  = find(len >= prctile(len, prctle));
        
        % weighted mean
        % get follicle coordinates
        x = msr.follicle_x(data_inds(ind));
        y = msr.follicle_y(data_inds(ind));
        nose_index = x > xnose & x < xrear_pad;
        %         weights  = 1./sqrt(((x(nose_index)-x0).^2 + (y(nose_index)-y0).^2));
        weights = ones(sum(nose_index), 1);
        ang_temp(k,1) = sum(weights.*(-msr.angle(data_inds(ind(nose_index)))))/sum(weights);                                                                                                                                                                                                                                                       
    end
    angle_notsmooth{f_ind} = ang_temp;
    ang{f_ind} = sgolayfilt(ang_temp, 4, 17); % 21frames:23.8Hz , 17frames:29.4Hz% Interpolate angle trace
    
    ang_interp = naninterp(ang{f_ind}, 'pchip');
    
    % whisking
    wsk{f_ind} = sgolayfilt(ang_temp, 4, 1001);
    
    % Calculate the phase
    dataFilt = genButterFilter(ang_interp, 1, 30, 4, 'butter_acausal', 500);
    yh       = hilbert(dataFilt);
    phs{f_ind}      = angle(yh);
    
    % Calculate set point and amplitude
    [envHigh, envLow] = envelope(ang_interp, 50, 'peak'); % originally 20A
    sp{f_ind}  = (envHigh + envLow)/2;
    amp{f_ind} = (envHigh - envLow)/2;
    
    % calculate the whisking velocity
    vel{f_ind} = [0; diff(ang_interp)/(1/500)];
    
end
progressbar(1)

% save
save([exp_dir filesep fname '-wt.mat'],...
    'angle_notsmooth', 'ang', 'phs', 'sp', 'amp', 'vel', 'wsk',...
    'face', '-v7.3')
disp(['completed processing: ' exp_name])
clear
