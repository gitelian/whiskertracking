% Convert seq high speed videos to an avi movie.
% The videos should contain a single painted whisker (white whisker on black
% background). This will invert the images so the whisker appears black on a
% flat white background. The tiff stack will be fed into the Janelia Farm
% whisker tracking software to see if it can extract more information about the
% whisker than the Hough Transform method used on this type of data.

function seq2avi(vid_dir, output_dir)
vid_path = '/media/greg/Data/HSV_Videos/FID0964/';
output_dir = '/media/greg/Data/HSV_Videos/FID0964_tiff/';
fstruct = dir([vid_path '*.seq']);

myfilter = fspecial('gaussian', [3 3], 0.5);

for vid = 1:length(fstruct) output_file_name = [fstruct(vid).name(1:end-4) '.avi'];
    [~, img] = Norpix2MATLAB([vid_path fstruct(vid).name]);
    img = uint8(img);
    tiff_img = zeros(size(img), 'uint8');
    writerObj = VideoWriter([output_dir output_file_name]);
    open(writerObj);
    for f = 1:size(img, 3);
        temp = img(:,:,f);
        temp = im2bw(temp, 0.05); %binarize image.
        temp = imfill(temp, 'holes'); %fill in gaps
        whisker_inds = find(temp == 1);
        background_inds = find(temp == 0);
        new_frame = zeros(size(temp), 'uint8');
        new_frame(whisker_inds) = 25;
        new_frame(background_inds) = 155;
        new_frame = imfilter(new_frame, myfilter, 'replicate');
        tiff_img(:, :, f) = new_frame;
        writeVideo(writerObj, new_frame);
    end
    close(writerObj);
end


