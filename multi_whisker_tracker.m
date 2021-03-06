%%MULTI_WHISKER_TRACKER Tracks pad angle from full pad high speed imaging
% run script to calculate smoothed angle, set-point, amplitude, phase, and
% velocity of whisker pad. This script expects tracking_data and .avi files
% for a given experiment.
%
% This will only work for an experiment with NO DROPPED FRAMES!!!
%
% TODO: add functionality to catch dropped frames and replace missing
% values with NaNs
%
% G. Telian
% Adesnik Lab
% UC Berkeley
% 20160921



%% get paths to measure, trace, and avi directories
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

% avi path
avi_dir = dir(['F:\wtavi' filesep exp_name filesep '*.avi']);
if isempty(avi_dir)
    error('avi directory empty!!!')
end

fname = msr_dir(1).name(1:7);

%% prompt user to select a point on the face of the mouse
prctle = 90;
fprintf('WAIT!!!\nTHIS MAY TAKE A WHILE TO LOAD\nYOU NEED TO PROVIDE A ROI\n')
msr = struct2table(LoadMeasurements([exp_dir filesep 'measure' filesep msr_dir(1).name]));
trc = struct2table(LoadWhiskers([exp_dir filesep 'trace' filesep trc_dir(1).name]));

v = VideoReader(['F:\wtavi' filesep exp_name filesep avi_dir(1).name]);
fig = figure('Position', [100, 100, v.Width/2, v.Height/2]);
ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;
k = v.NumberOfFrames;
vidFrame = read(v, k);
image(vidFrame);
currAxes.Visible = 'off';
hold on

% get indices for data corresponding to frame k
data_inds = find(msr.fid == k-1);
% plot all whisker traces
for l = data_inds(1):data_inds(end);
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
    
[x0, y0] = getpts(fig);

x0 = x0(1);
y0 = y0(1);
close all
clear v trc msr
% x0 = 1375;
% y0 = 550;


%% measure angle and calculate smooth angle trace
disp('smoothing angle trace')
ang = [];
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
        dist_square  = 1./((x-x0).^2 + (y-y0).^2);
        ang_temp(k,1) = sum(dist_square.*(-msr.angle(data_inds(ind))))/sum(dist_square);                                                                                                                                                                                                                                                       
    end
    ang = [ang; ang_temp];
end
progressbar(1)
ang_smooth = sgolayfilt(ang, 4, 21);

%% calculate smoothed angle, set-point, amplitude, velocity, phase

num_frames            = length(ang_smooth);
angle_mat             = ang_smooth;
angle_mat_interp      = nan(num_frames,1);
phase_mat             = nan(num_frames,1);
angle_mat_interp_filt = nan(num_frames,1);
set_point_mat         = nan(num_frames,1);
amp_mat               = nan(num_frames,1);

% Interpolate angle trace
disp('interpolating')
angle_mat_interp(:,1) = naninterp(angle_mat(:,1),'pchip');

% Calculate the phase of the signal between 1 and 30Hz
% THE OG dataFilt = genButterFilter(angle_mat_interp(:,1),4,100,4,'butter_acausal',500);
disp('phase')
dataFilt = genButterFilter(angle_mat_interp(:,1),1,30,4,'butter_acausal',500);
yh = hilbert(dataFilt);
phase_mat(:,1) = angle(yh);
angle_mat_interp_filt(:,1) = dataFilt;

% Calculate set point
% get the set point by low pass filtering angle trace, finding peaks
% and troughs, getting midpoint between them, and smoothing points with
% a 50 point sliding window
disp('set-point')
dataFilt = genButterFilter(angle_mat_interp(:,1),5,20,4,'butter_acausal',500);
[~,pkLoc] = findpeaks(dataFilt,'MinPeakHeight',0,'MinPeakDistance',5);
[~,trLoc] = findpeaks(-dataFilt,'MinPeakDistance',5);

temp = nan(num_frames,2);
temp(pkLoc,1)      = angle_mat_interp(pkLoc,1);
temp(trLoc,2)      = angle_mat_interp(trLoc,1);
temp(1,1)          = angle_mat_interp(1,1);
temp(1,2)          = angle_mat_interp(1,1);
temp(num_frames,1) = angle_mat_interp(num_frames,1);
temp(num_frames,2) = angle_mat_interp(num_frames,1);
temp(:,1)          = naninterp(temp(:,1),'spline');
temp(:,2)          = naninterp(temp(:,2),'spline');
mid                = nan(num_frames,1);
mid(:,1)           = smooth(mean(temp,2),50);

% Calculate the amplitude of the whisking envelope
disp('amplitude')
dataFilt = genButterFilter(angle_mat_interp(:,1),5,50,4,'butter_acausal',500);
[~,pkLoc] = findpeaks(dataFilt,'MinPeakHeight',0,'MinPeakDistance',5);
[~,trLoc] = findpeaks(-dataFilt,'MinPeakDistance',5);
amp = nan(num_frames,2);
amp(pkLoc,1)      = angle_mat_interp(pkLoc,1);
amp(trLoc,2)      = angle_mat_interp(trLoc,1);
amp(1,1)          = angle_mat_interp(1,1);
amp(1,2)          = angle_mat_interp(1,1);
amp(num_frames,1) = angle_mat_interp(num_frames,1);
amp(num_frames,2) = angle_mat_interp(num_frames,1);
amp(:,1)          = naninterp(amp(:,1),'spline');
amp(:,2)          = naninterp(amp(:,2),'spline');

set_point_mat(:,1) = mid;
amp_mat(:,1)       = (amp(:,1) - amp(:,2))/2;

% calculate the whisking velocity
vel_mat = diff(angle_mat)/(1/500);

% !!! add appropriate path and experiment name !!!
save([exp_dir filesep fname '-wt.mat'], 'ang_smooth', 'angle_mat',...
    'set_point_mat', 'amplitude', 'velocity', 'phase_mat', '-v7.3')



% %% plotting
% %  plot raw mean angle with smoothed mean angle
% t = linspace(0, length(ang_smooth)/500, length(ang_smooth));
% plot(t, ang,'k');hold on;
% plot(t, ang_smooth,'r')
% xlabel('time (s)')
% ylabel('angle (deg)')
% 
% t = linspace(0, num_frames/500, num_frames);
% % set-point
% subplot(1,2,1)
% plot(t, angle_mat, 'k', t, set_point_mat, 'r')
% xlim([0 3.5])
% ylim([90 180])
% 
% % phase
% subplot(1,2,2)
% plot(t, angle_mat,'k', t, phase_mat*2+135,'-r')
% xlim([0 1.5])
% ylim([90 180])
% 
% % amplitude
% plot(t, angle_mat, 'k', t, amp_mat+set_point_mat, 'r', t, -amp_mat+set_point_mat, 'r')


