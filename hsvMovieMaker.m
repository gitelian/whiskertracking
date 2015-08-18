% HSVMOVIEMAKER Make a movie of angle, set-point, phase, and run-speed traces
% for the specified high speed video.


fid     = '1034';
trialNum   = 13;
whiskerNum = 0;
fps        = 500; % frames per second
sr         = 30000;
startTime  = 0.5;
stopTime   = 2.5;
sr         = 30000;

% Load MCdata for running
fstruct = dir(['~/Documents/AdesnikLab/Data/' fid '*.dat'])
load(['~/Documents/AdesnikLab/Data/' fstruct.name], '-mat')
[trialsran, trialswalk, runspeed, time] = classify_run_trials(run_data, 30000,...
250, 25, 0.25, 2.0, 0);
clear MCdata

% Load whisker measurements
fstruct = dir(['~/Documents/AdesnikLab/Processed_HSV/FID' fid '*.mat'])
load(['~/Documents/AdesnikLab/Processed_HSV/' fstruct.name] , '-mat')

timeStep  = round(1/fps*sr)/sr;
numOfNans = round(0.25*fps);
tMov = linspace(startTime-0.250,(camTime(end) + 0.250 - 1/fps),numFrames+2*numOfNans)';
runspeedMat = downsample(runspeed{trialNum}, floor(length(runspeed{trialNum})/...
    length(angleCell{trialNum,1})));

angleMatInterp = [nan(numOfNans,1);angleCell{trialNum,1};nan(numOfNans,1)];
setpointMat    = [nan(numOfNans,1);setPointCell{trialNum,1};nan(numOfNans,1)];
phaseMat       = [nan(numOfNans,1);phaseCell{trialNum,1};nan(numOfNans,1)];
velocity       = [nan(numOfNans,1);runspeedMat;nan(numOfNans,1)]*(2*pi*6/360);

hsvmovietraces(trialNum,numFrames,tMov,numOfNans,angleMatInterp,setpointMat,...
    phaseMat,velocity)
