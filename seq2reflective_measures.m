% SEQ2REFLECTIVE_MEASURES Function that converts high speed video of whiskers
% to 2d histogram.
% Extract angle information from a single reflective whisker.
% IT IS VERY IMPORTANT THAT ONLY ONE WHISKER IS REFLECTIVE AND THAT THE
% REFLECTIVE PORTION IS LINEAR!
%
% Input: path to seq high speed movie file
%
% Output
% angles: vector of whisker angles for each frame.
% mean_counts: 2d histogram (probability) values for movie.
% lines: cell array that contains line objects for each frame computed using
% a Hough transform.
% img: 3d matrix of image values. ij correspond to pixels and the third
% dimension corresponds to frames.

function [angles, mean_counts, img] = seq2reflective_measures(file_path, roi, follicle);
[~, img] = Norpix2MATLAB(file_path);
[~, img] = Norpix2MATLAB('/media/greg/Data/HSV_Videos/FID0964/FID964_15-55-19.027.seq');
img = uint8(img);
temp = reshape(img, size(img,1)*size(img,2)*size(img,3), 1);

% Find threshold for white pixels
hist_threshold = mean(temp) + 4*std(single(temp));

clear temp

% Calculate 2d histogram for entire movie.
counts = zeros(size(img,1), size(img,2));
for frind = 1:size(img,3)
    temp = img(:,:,frind);
    img_inds = temp > hist_threshold;
    counts(img_inds) = counts(img_inds) + 1;
end

mean_counts = counts/size(img,3);

% Use ROI and follicle position to compute angle
se = strel('disk', 20);
angles = nan(size(img,3),1);
parfor f = 1:size(img, 3)
    temp = img(:,:,f);
    BW = im2bw(temp, 0.1);
    BW_blobs = imclose(BW, se);
    BW_and_roi = BW_blobs & roi;
    BW_connected_components = bwconncomp(BW_and_roi);
    BW_measurements = regionprops(BW_connected_components, 'Area');
    blob_areas = [BW_measurements.Area];
    [~, indexOfBiggestBlob] = max(blob_areas);
    blob_centroids = regionprops(BW_blobs, 'centroid');
    if ~isempty(indexOfBiggestBlob)
        xy1 = round(blob_centroids(indexOfBiggestBlob).Centroid);
        xy2 = follicle;
        % repeated code because MATLAB does not deal with temporary variables
        % in a reasonable way. You can't put xy_new outside of the if statement
        % because it can be deleted before being assigned. MATLAB is stupid.
        if xy1(2) > xy2(2)
            xy_new = xy1 - xy2;
            ang = 180 - atan2(xy_new(2), xy_new(1))*180/pi;
            angles(f,1) = ang;
        elseif xy1(2) < xy2(2)
            xy_new = xy2 - xy1;
            ang = 180 - atan2(xy_new(2), xy_new(1))*180/pi;
            angles(f,1) = ang;
        end
    end
end
