trc = struct2table(LoadWhiskers('FID1295-12-13-24.673.avi.whiskers'));
msr = struct2table(LoadMeasurements('FID1295-12-13-24.673.avi.measurements'));
v = VideoReader('FID1295-12-13-24.673.avi');
newvid = VideoWriter('FID1295-whole-trial.avi', 'Motion JPEG AVI');
open(newvid);

%% measure angle and calculate smooth angle trace
ang = zeros(v.NumberOfFrames,1);
for k = 1:v.NumberOfFrames
    % get indices for data corresponding to frame k
    data_inds = find(trc.time == k-1);
    % get length of whisker traces for frame k
    % sort whisker traces by longest to shortes    
    wids = msr.wid(data_inds);
    len  = msr.length(data_inds);
    ind  = find(len >= prctile(len, 95));
    ang(k,1) = mean(-msr.angle(data_inds(ind)));
end
ang_smooth = sgolayfilt(ang, 3, 11);

%% plot whisker traces overlay and whisker angle approximation overlay
fig = figure('Position', [100, 100, v.Width/2, v.Height/2]);
ColOrd = get(gca,'ColorOrder');ColOrd(2,2)=0.75;

for k = 1:v.NumberOfFrames
    
    % Plot all whisker traces in a single color    
    % plot frame k
    vidFrame = read(v, k);
    image(vidFrame);
    currAxes.Visible = 'off';
%     
%     hold on
%     % get indices for data corresponding to frame k
%     data_inds = find(trc.time == k-1);
% 
%     % plot all whisker traces
%     for l = data_inds(1):data_inds(end);
%         plot(trc.x{l}, trc.y{l})
%         title(num2str(l))
% %         pause(1)
%     end
%     hold off

%     % get length of whisker traces for frame k
%     % sort whisker traces by longest to shortes    
%     wids = msr.wid(data_inds);
%     len  = msr.length(data_inds);
%     ind  = find(len >= prctile(len, 95));
% %     [~, ind] = sort(len, 'descend'); % index of longest to shortest whiskers
% 
%     % calculate and plot angle
%     ang = mean(-msr.angle(data_inds(ind)));
%     x0  = 1300 - 150*cos((180 - ang)*pi/180);
%     y0  = 600  - 150*sin((180 - ang)*pi/180);
%     plot([x0, 1300], [y0, 600], 'r')
%     
%     x1  = 1300 - 150*cos((180 - ang_smooth(k))*pi/180);
%     y1  = 600  - 150*sin((180 - ang_smooth(k))*pi/180);    
%     plot([x1, 1300], [y1, 600], 'g')
%     
%     % plot longest to shortest
%     for l = 1:length(ind)
%         wid = wids(ind(l));
%         trc_ind = find(table2array(trc(data_inds, 'id')) == wid);
%         plot(trc.x{data_inds(trc_ind)}, trc.y{data_inds(trc_ind)})
% %         pause(0.1)
%     end
% %     pause(1)
    

    
%%  grab frame and add it to the movie
    disp(k);
    F = getframe(fig);
    writeVideo(newvid,F);
end

%%
close(newvid)
%% plotting
t = linspace(0, length(ang)/500, length(ang));
plot(t, ang,'k');hold on;
plot(t, ang_smooth,'r')
xlabel('time (s)')
ylabel('angle (deg)')
