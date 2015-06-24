% Convert high speed videos of whiskers to 2d histograms.
% The histograms can be used later to extract overall setpoint and amplitude.
%
% Scratch script for testing.

vid_dir = '/media/greg/Data/HSV_Videos/FID0964';
vid_fname = 'FID0961_2015-05-06_13-04-48.814.seq';
fstruct = dir([vid_dir filesep '*.seq']);
vid_fname = fstruct(15).name;

tic()
white_background = 0;

[~, img] = Norpix2MATLAB([vid_dir filesep vid_fname]);
img = uint8(img);
temp = reshape(img, size(img,1)*size(img,2)*size(img,3), 1);

if white_background
    threshold = median(temp)*3/4;
else
    threshold = mean(temp) + 4*std(single(temp));
end

clear temp

counts = zeros(size(img,1), size(img,2));

for frind = 1:size(img,3)
    temp = img(:,:,frind);
    if white_background
        img_inds = temp < threshold;
    else
        img_inds = temp > threshold;
    end
    counts(img_inds) = counts(img_inds) + 1;
end

mean_counts = counts/size(img,3);

% For regular colors
figure
imagesc(mean_counts, [0,0.1])
colormap hot

% For inverted colors
figure
imagesc(abs(1 - mean_counts), [0.9,1])
colormap gray

% Use Hough transform to get whisker trace
angles = nan(size(img,3),1);
lines  = cell(size(img,3),1);
parfor f = 1:size(img, 3)
    temp = img(:,:,f);
    temp = im2bw(temp, 0.2);
    temp = imfill(temp, 'holes');
    BW = edge(temp, 'canny');
    [H, theta, rho] = hough(BW);
    P = houghpeaks(H,1,'threshold', ceil(0.3*max(H(:))));
    l = houghlines(BW, theta, rho, P, 'FillGap', 150, 'MinLength', 100);
    if ~isempty(l)
        xy1 = l.point1;
        xy2 = l.point2;
        if xy1(2) > xy2(2)
            xy_new = xy1 - xy2;
        elseif xy1(2) < xy2(2)
            xy_new = xy2 - xy1;
        end
        ang = 180 - atan2(xy_new(2), xy_new(1))*180/pi;
        angles(f,1) = ang;
        lines{f,1} = l;
    end
end
toc()
t = 0:size(img, 3) - 1;
figure;
plot(t, angles);
ylim([90 180])

% Code that attempts to find the sector boundaries of a 2d histogram whisker
% trace. Play around with this IF automated computing of sector boundaries is
% important (i.e. if we want to calculate the mean set-point across all trials.
% Maybe there is an easier way: draw a small rectangular roi (or line) and
% calculate the region boundary using three points. The first point is the max
% protraction point, second point is the min protraction point, and the third
% point is a user define point placed on base of the whisker follicle. Using
% these three points I can calculate the sector angle (amplitude), and
% set-point.
%
%BW1 = im2bw(mean_counts, 1/6*(threshold/255));
%BW = imfill(BW1, 'holes');
%figure;imagesc(BW);colormap gray
%BW = edge(BW, 'canny');
%[H,theta,rho] = hough(BW);
%P = houghpeaks(H,2,'threshold',ceil(0.3*max(H(:))));
%lines = houghlines(BW,theta,rho,P,'FillGap',150,'MinLength',100);
%figure;colormap gray
%imagesc(abs(1 - mean_counts), [0.9,1])
%hold on
%max_len = 0;
%for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
%
%    % Determine the endpoints of the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2);
%    if ( len > max_len)
%        max_len = len;
%        xy_long = xy;
%    end
%end
%hold off
%toc()
