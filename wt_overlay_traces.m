
%% get paths to measure, trace, and avi directories

mov = 1;
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

% avi path
avi_dir = dir(['F:\wtavi' filesep exp_name filesep '*.avi']);
if isempty(avi_dir)
    error('avi directory empty!!!')
end

% movie overlay path
movie_dir = [exp_dir filesep 'movies'];
if ~exist(movie_dir, 'dir')
    mkdir(movie_dir)
end

fname = msr_dir(1).name(1:7);

%%
msr = struct2table(LoadMeasurements([exp_dir filesep 'measure' filesep msr_dir(mov).name]));
trc = struct2table(LoadWhiskers([exp_dir filesep 'trace' filesep trc_dir(mov).name]));
load([exp_dir filesep fname '-wt.mat'])

v = VideoReader(['F:\wtavi' filesep exp_name filesep avi_dir(mov).name]);
fig = figure('Position', [100, 100, v.Width*2, v.Height*2]);
ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;

newvid = VideoWriter([exp_dir filesep 'movies' filesep avi_dir(mov).name], 'Motion JPEG AVI');
k = v.NumberOfFrames;
vidFrame = read(v, 1);
image(vidFrame);
currAxes.Visible = 'off';
hold on
% get indices for data corresponding to frame k
data_inds = find(msr.fid == 0);
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
    
[x0, y0] = getpts(fig);
% % define ROI
% [~, xv, yv] = roipoly(fig);

x0 = x0(1);
y0 = y0(1);
close all

%% plot whisker traces overlay and whisker angle approximation overlay
open(newvid);
%%
fig = figure('Position', [100, 100, v.Width*2, v.Height*2]);
ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;
count = 1; %+(mov-1)*v.NumberOfFrames;
num_frames = length(unique(msr.fid));
for k = 1:num_frames %v.NumberOfFrames
    
    % Plot all whisker traces in a single color    
    % plot frame k
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
%         pause(1)
    end

    % get length of whisker traces for frame k
    % sort whisker traces by longest to shortes    
    wids = msr.wid(data_inds);
    len  = msr.length(data_inds);
    ind  = find(len >= prctile(len, prctle));

    % get follicle coordinates
    x = msr.follicle_x(data_inds(ind));
    y = msr.follicle_y(data_inds(ind));
%     in = inpolygon(x,y,xv,yv);
    plot(x, y, 'co');
    
    x1  = x0 - 100*cos((180 - ang{mov}(count))*pi/180);
    y1  = y0  - 100*sin((180 - ang{mov}(count))*pi/180);
%     if ang(count) < 0
%         x1  = x0 + 100*cos((180 + ang(count))*pi/180);
%         y1  = y0 + 100*sin((180 + ang(count))*pi/180);
%     end
%     x1  = mean(x) - 100*cos((180 - ang(count))*pi/180);
%     y1  = mean(y)  - 100*sin((180 - ang(count))*pi/180);
    count = count + 1;
    
    plot([x1, x0], [y1, y0], '-go')
%     plot([x1, mean(x)], [y1, mean(y)], '-go')
    
    % plot subset from longest to shortest
    for l = 1:length(ind)
        wid = wids(ind(l));
        trc_ind = find(table2array(trc(data_inds, 'id')) == wid);
        plot(trc.x{data_inds(trc_ind)}, trc.y{data_inds(trc_ind)},'--r')
%         pause(0.025)
    end
    text(v.Width*0.75, v.Height*0.90,['Frame: ' num2str(k)],'Color','white','FontSize',14)

%%  grab frame and add it to the movie
%     disp(k);
    F = getframe(fig);
    writeVideo(newvid,F);
    pause(0.05)
    
    if mod(k, 11) == 0
        close all
        fig = figure('Position', [100, 100, v.Width*2, v.Height*2]);
        ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;
        %pause(0.5)
    end
end

close(newvid)
close(fig)
clear
