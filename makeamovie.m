



function makeamovie(fid,trialNum)

VideoFilePath = ['~/Documents/AdesnikLab/HSV_Data/' fid];
dirstruct = dir([VideoFilePath filesep '*.seq']);

video = VideoWriter(['Trial-' num2str(trialNum) 'movie.avi'],'Motion JPEG AVI');
set(video,'FrameRate',30);
open(video)
[~, Imgs] = Norpix2MATLAB([VideoFilePath filesep dirstruct(trialNum).name]); % load movie
load(['~/Documents/AdesnikLab/Processed_HSV/' fid '-data-cells.mat']) % load tracking data
h = figure;

for frame = 1:size(Imgs,3)
    Img = Imgs(:,:,frame);
    figure(h);
    imagesc(Img);
    axis off square
    colormap gray
    hold all
    plot(wcoordCell{trialNum}{frame}(:,1), wcoordCell{trialNum}{frame}(:,2), '-ro');
    text(1450,700,['time (s): ' sprintf('%.3f', camTime(frame))],'BackgroundColor',[.9 .9 .9])
    hold off
    img = getframe(gca);
    writeVideo(video,img);
end
video.close;
end
