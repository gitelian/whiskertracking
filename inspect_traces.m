% inspect_traces loads in whisker tracking data and plots the angle, set point,
% bad indicies, as well as the censor period that should not be analyzed.

fid = 'FID1067'
load(['~/Documents/AdesnikLab/Processed_HSV/' fid '-data-cells.mat'], '-mat')

for k = 1:length(angleMat)
    angle = angleMat(:, k);
    setPoint = setPointMat(:, k);
    censor = zeros(length(camTime), 1);
    censorCell{k}(censorCell{k} <= 0) = [];
    censor(censorCell{k}) = 155;
    bad_frames = zeros(length(camTime), 1);
    bad_frames(badIndsCell{k}) = 150;

    h = figure;
    plot(angle);
    hold on
    plot(setPoint, 'r')
    plot(censor, 'g')
    plot(bad_frames, 'm')
    hold off
    ylim([50 180])
    title([fid '_Trial' sprintf('%02d',k)])
    saveas(h, ['~/Documents/AdesnikLab/Figures/HSV_Reflective/' fid '_Trial' sprintf('%02d',k) '.pdf'])
    close(h)
end
