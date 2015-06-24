% track_reflective_whisker takes no arguments. It prompts the user to select a
% high speed video directory to analyze and to enter in some details about the
% experiment such as when the video began recording, when it stopped, when
% the object started moving (i.e. stim period began), and the frame rate of
% the camera. The processed data will be saved in the directory:
% 'E:\TrackingData\ProcessedMeasurements\' and will contain cell arrays
% with angle, curvature, phase, and other information for each trial. All
% vectors contained in the cell are the same lenghth and can be easily
% indexed during analysis.
%
% G. Telian
% Adesnik Lab
% UC Berkeley
% 20140611
%
% 20140625 update (GT): changed setpoint code to get a more accurate measure of
% setpoint.
%
% 20150514 update (GT): changed filter settings to get a more accurate
% measurement of phase. Also, adapted code to process videos of a single
% reflective whisker (painted with reflected paint).

function track_reflective_whisker()

tic()

% Have the user verify/change video paramenters
prompt    = {'start time','stop time','object start time','frames per second',...
    'e-phys sampling rate','Total frames'};
dlgTitle  = 'High speed video parameters';
numLines  = 1;
default   = {'0.5','2.5','1.0','500','30000','1000'};
usrInput  = inputdlg(prompt,dlgTitle,numLines,default);

if isempty(usrInput)
    disp('process canceled')
    return
end

startTime = str2double(usrInput{1});
stopTime  = str2double(usrInput{2});
objTime   = str2double(usrInput{3});
fps       = str2double(usrInput{4});
sr        = str2double(usrInput{5});
numFrames = str2double(usrInput{6});

timeStep  = round(1/fps*sr)/sr;

% Have user specify the measure directory
vidDir = uigetdir('E:\TrackingData\Measure','Select the video directory to analyze');

%% REMOVE THIS %%
%vidDir = '/media/greg/Data/HSV_Videos/FID0964'

[~,fileName] = fileparts(vidDir);

% List Directories Contents
vidStrct = dir([vidDir filesep '*.seq']);

if vidDir == 0
    error('measure directory not selected')
elseif isempty(vidStrct)
    error('measure directory is empty!')
end

% Get common (fid) directory name
[~,comDir,~] = fileparts(vidDir);

%% Extract Angle and Curvature Data for Each Tracked Whisker

numMovies     = length(vidStrct);
angleCell     = cell(numMovies,1);
badIndsCell   = cell(numMovies,1);
censorCell    = cell(numMovies,1);
imhistCell    = cell(numMovies,1);
phaseCell     = cell(numMovies,1);
angleZeroMean = cell(numMovies,1);
setPointCell  = cell(numMovies,1);
ampCell       = cell(numMovies,1);
baseFrame     = floor((objTime-startTime)/timeStep); %Frame number when object starts moving (i.e. end of baseline period)
camTime       = startTime:timeStep:(stopTime-timeStep);

stimFrame     = floor((1.5-startTime)/timeStep);


for trialNum = 1:numMovies

    fName = vidStrct(trialNum).name;

    angleMat           = nan(numFrames,1);
    angleMatInterp     = nan(numFrames,1);
    phaseMat           = nan(numFrames,1);
    angleMatInterpFilt = nan(numFrames,1);
    setPointMat        = nan(numFrames,1);
    ampMat             = nan(numFrames,1);

    if trialNum == 1
        [~, imgOut] = Norpix2MATLABopenSingleFrame([vidDir filesep fName],1);
        h1 = figure;
        imshow(imgOut);
        roi = roipoly;
        close(h1);
        h1 = figure;
        imshow(imgOut);
        follicle = ginput(1);
        close all;
        h = waitbar(0, 'Tracking Progress...');
    end

    [angleMat, img2dhist, ~] = seq2reflective_measures([vidDir filesep fName],...
        roi, follicle);

    upper_thresh = nanmean(angleMat) + 3*nanstd(angleMat);
    lower_thresh = nanmean(angleMat) - 3*nanstd(angleMat);
    angleMat(angleMat(:,1) <= lower_thresh,1) = nan; %removes discontinuities
    angleMat(angleMat(:,1) >= upper_thresh,1) = nan;

    % Find bad indices and censory periods (i.e. regions not to be analyzed
    % because of poor or nonexistent data
    bad_inds = find(isnan(angleMat) == 1);
    censor_mat = [];
    for k = 1:length(bad_inds)
        bad_i = bad_inds(k);
        lower_thresh = bad_i-25;
        upper_thresh = bad_i+25;
        num_bad_points = length(find(bad_inds <= upper_thresh & bad_inds >= lower_thresh));
        if num_bad_points >= 25
            censor_mat = [censor_mat, (lower_thresh:upper_thresh)'];
        end
    end
    censor_mat = unique(censor_mat);
    censor_mat(censor_mat <= 0) = [];
    censor_mat(censor_mat > numFrames) = [];

    if length(bad_inds) < 0.5*length(angleMat)

        % Interpolate angle trace and remove discontinuities
        angleMatInterp(:,1) = smooth(naninterp(angleMat(:,1),'pchip'),3);

        angleMatInterp(angleMatInterp(:,1)<= 70,1) = 90; %removes discontinuities
        angleMatInterp(angleMatInterp(:,1)>=200,1) = 180;

        % Calculate the phase of the signal between 15 and 25Hz
        % THE OG dataFilt = genButterFilter(angleMatInterp(:,1),4,100,4,'butter_acausal',500);
        dataFilt = genButterFilter(angleMatInterp(:,1),15,25,4,'butter_acausal',500);
        yh = hilbert(dataFilt);
        phaseMat(:,1) = angle(yh);
        angleMatInterpFilt(:,1) = dataFilt;

        % Calculate set point
        % get the set point by low pass filtering angle trace, finding peaks
        % and troughs, getting midpoint between them, and smoothing points with
        % a 50 point sliding window
        dataFilt = genButterFilter(angleMatInterp(:,1),5,20,4,'butter_acausal',500);
        [~,pkLoc] = findpeaks(dataFilt,'MinPeakHeight',0,'MinPeakDistance',5);
        [~,trLoc] = findpeaks(-dataFilt,'MinPeakDistance',5);

        temp = nan(numFrames,2);
        temp(pkLoc,1)     = angleMatInterp(pkLoc,1);
        temp(trLoc,2)     = angleMatInterp(trLoc,1);
        temp(1,1)         = angleMatInterp(1,1);
        temp(1,2)         = angleMatInterp(1,1);
        temp(numFrames,1) = angleMatInterp(numFrames,1);
        temp(numFrames,2) = angleMatInterp(numFrames,1);
        temp(:,1)         = naninterp(temp(:,1),'spline');
        temp(:,2)         = naninterp(temp(:,2),'spline');
        mid               = nan(numFrames,1);
        mid(:,1)          = smooth(mean(temp,2),50);

        % Calculate the amplitude of the whisking envelope
        dataFilt = genButterFilter(angleMatInterp(:,1),5,50,4,'butter_acausal',500);
        [~,pkLoc] = findpeaks(dataFilt,'MinPeakHeight',0,'MinPeakDistance',5);
        [~,trLoc] = findpeaks(-dataFilt,'MinPeakDistance',5);
        amp = nan(numFrames,2);
        amp(pkLoc,1)     = angleMatInterp(pkLoc,1);
        amp(trLoc,2)     = angleMatInterp(trLoc,1);
        amp(1,1)         = angleMatInterp(1,1);
        amp(1,2)         = angleMatInterp(1,1);
        amp(numFrames,1) = angleMatInterp(numFrames,1);
        amp(numFrames,2) = angleMatInterp(numFrames,1);
        amp(:,1)         = naninterp(amp(:,1),'spline');
        amp(:,2)         = naninterp(amp(:,2),'spline');

    %     figure;plot(camTime,mid(:,1),'k',camTime,angleMatInterp(:,1),'b');
    %     figure;plot(camTime,angleMatInterp(:,1),camTime,amp(:,1),camTime,amp(:,2),camTime,mid,'k')
        setPointMat(:,1) = mid;
        ampMat(:,1)      = (amp(:,1) - amp(:,2))/2;

        angleCell{trialNum,1}     = angleMatInterp;
        badIndsCell{trialNum,1}   = bad_inds;
        censorCell{trialNum,1}    = censor_mat;
        imhistCell{trialNum,1}    = img2dhist;
        phaseCell{trialNum,1}     = phaseMat;
        angleZeroMean{trialNum,1} = angleMatInterpFilt;
        setPointCell{trialNum,1}  = setPointMat;
        ampCell{trialNum,1}       = ampMat;

    else
        goodTrials(trialNum, 1)   = 0;
        angleCell{trialNum,1}     = angleMatInterp;
        badIndsCell{trialNum,1}   = bad_inds;
        censorCell{trialNum,1}    = censor_mat;
        imhistCell{trialNum,1}    = img2dhist;
        phaseCell{trialNum,1}     = phaseMat;
        angleZeroMean{trialNum,1} = angleMatInterpFilt;
        setPointCell{trialNum,1}  = setPointMat;
        ampCell{trialNum,1}       = ampMat;
    end


%    disp([num2str(trialNum) '/' num2str(numMovies)])
fracComplet = trialNum/numMovies;
waitbar(fracComplet,h,sprintf('%d of %d complete',[trialNum numMovies]))
end

close(h)

save(['E:\TrackingData\ProcessedMeasurements\'...
    comDir '-data-cells.mat'],'angleCell','badIndsCell','censorCell',...
    'imhistCell','phaseCell', 'angleZeroMean','setPointCell','ampCell','numFrames',...
    'camTime','startTime','stopTime','fps','-v7.3')

%save(['~/Documents/AdesnikLab/Processed_HSV/'...
%    comDir '-data-cells.mat'],'angleCell','badIndsCell','censorCell',...
%    'imhistCell','phaseCell', 'angleZeroMean','setPointCell','ampCell','numFrames',...
%    'camTime','startTime','stopTime','fps','-v7.3')

toc()



