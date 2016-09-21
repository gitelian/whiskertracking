

%% measure angle and calculate smooth angle trace
msr_dir = dir('measure_mat_files/*-table.mat');
% prompt user to select a point on the face of the mouse
prctle = 90;
x0 = 1375;
y0 = 550;
disp('smoothing angle trace')
ang = [];
progressbar('files', 'frames')
for f_ind = 1:length(msr_dir)
    progressbar(f_ind/length(msr_dir), [])
    load(['measure_mat_files/' msr_dir(f_ind).name],'-mat');
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
save('wt.mat', 'ang_smooth', 'angle_mat', 'set_point_mat', 'amplitude', 'velocity', 'phase_mat', '-v7.3')



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


