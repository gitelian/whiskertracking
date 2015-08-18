
num_trials = size(ampCell, 1);

%% Convert ampCell to ampMat
%  rows: frames, cols: trials
temp = reshape(ampCell, 1, num_trials);
ampMat = cell2mat(temp);
clear ampCell

%% Convert angleCell to angleMat
temp = reshape(angleCell, 1, num_trials);
angleMat = cell2mat(temp);
clear angleCell

%% Convert angleZeroMean from cell to matrix
temp = reshape(angleZeroMean, 1, num_trials);
angleZeroMean = cell2mat(temp);

%% Convert imhistCell to imhistMat
% temp = zeros(size(imhistCell{1}, 1), size(imhistCell{2}, 2), num_trials);
% for k = 1:num_trials
%     temp(:, :, k) = imhistCell{k};
% end
% imhistMat = temp;
% clear imhistCell

%% Convert phaseCell to phaseMat
temp = reshape(phaseCell, 1, num_trials);
phaseMat = cell2mat(temp);
clear phaseCell

%% Convert setPointCell to setPointMat
temp = reshape(setPointCell, 1, num_trials);
setPointMat = cell2mat(temp);
clear setPointCell

clear temp

save('FID1038-data-cells.mat', '-v7.3')