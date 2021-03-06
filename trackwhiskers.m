%%TRACKWHISKERS.M Tracks whiskers using Janelia farm tracker
%   https://openwiki.janelia.org/wiki/display/MyersLab/Whisker+Tracking
%   TRACKWHISKERS takes no arguments; it prompts the user to select a
%   directory contining .avi files of high speed videos.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20160815                                   



% Have user specify the video directory
vidDir = uigetdir('~/Documents/avi_hsv', 'Select the video directory to analyze');
[~, dirName,~] = fileparts(vidDir);
% dirCell = {'FID734'};
% whiskerMat = [1];

prompt   = {'Number of whiskers','Face position'};
dlgTitle = 'Whisker Tracker Parameters';
numLines = 1;
%default  = {'-1','1650','680'};
default  = {'-1','Face pos: top, bottom, left, right'}; % -1 means 'classify' will figure out the length threshold for classification
usrInput = inputdlg(prompt,dlgTitle,numLines,default);

if isempty(usrInput)
    disp('process canceled')
    return
end

whiskerMat = str2double(usrInput{1});
faceX      = usrInput{2};
%faceY      = usrInput{3};

mov_path = '/home/beast/Documents/avi_hsv';
mes_path = ['/home/beast/Documents/tracking_data/' dirName filesep 'measure'];
trc_path = ['/home/beast/Documents/tracking_data/' dirName filesep 'trace'];

fileMap  = [];
for session = 1:1;%length(dirName)

%     dirName = dirName{session};
    num_whiskers = whiskerMat(session);
    
%      fstruct = dir([mov_path filesep dirName filesep '*.seq']);
    fstruct = dir([mov_path filesep dirName filesep '*.avi']);
    
    if exist(mes_path,'dir') ~= 7
        mkdir(mes_path)
    end
    
    if exist(trc_path,'dir') ~= 7
        mkdir(trc_path)
    end
    
    tic
    
    %% Parallel Process
    parfor k = 1:length(fstruct)
        fname = fstruct(k).name;
        system(['~/Documents/whisk-1.1.0d-Linux/bin/whisk/'...
            'trace ' mov_path filesep dirName filesep fname ' ' trc_path filesep fname '.whiskers'])
    end
    
    %% Measure/Classify/Reclassify Trace Files
    fileCount   = 0;
    filesMissed = 0;
    for k = 1:length(fstruct)
        
        fname = fstruct(k).name;
        
        status = system(['~/Documents/whisk-1.1.0d-Linux/bin/whisk/'...
            'measure --face ' faceX ' ' trc_path filesep fname '.whiskers' ' '...
            mes_path filesep fname '.measurements']);
        
        status = system(['~/Documents/whisk-1.1.0d-Linux/bin/whisk/'...
            'classify ' mes_path filesep fname '.measurements' ' '...mov
            mes_path filesep fname '.measurements ' faceX ' ' '--px2mm 0.014 -n '...
            num2str(num_whiskers)]); % mm/px = 0.0198
        
        %--limit0.01:500']); % mm/px = 0.0198
        
        status = system(['~/Documents/whisk-1.1.0d-Linux/bin/whisk/'...
            'reclassify ' mes_path filesep fname '.measurements' ' '...
            mes_path filesep fname '.measurements' ' -n ' num2str(num_whiskers)])
%         
        msrName = [mes_path filesep fname '.measurements'];
        
        disp(' ')
        disp(['File number: ' num2str(k)])
        
        if exist(msrName,'file') == 2
            fileCount = fileCount + 1;
            fileMap(k,session) = fileCount;
        elseif exist(msrName,'file') == 0
            filesMissed = filesMissed + 1;
            fileMap(k,session) = 0;
        end
        
        disp(['Files missed: ' num2str(filesMissed)])
        disp(' ')
        
    end
    
    save(['~/Documents/tracking_data/' dirName filesep 'fileMap.mat'],'fileMap','-v7.3')
    
end

b = toc;

disp(['total time ' num2str(b/(3600*24)) 'days'])
clear all





