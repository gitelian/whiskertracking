%%TRACKWHISKERS.M Tracks whiskers using Janelia farm tracker
%   https://openwiki.janelia.org/wiki/display/MyersLab/Whisker+Tracking
%   TRACKWHISKERS takes no arguments; it prompts the user to select a
%   directory contining .avi files of high speed videos.
%
%   G. Telian

%   Adesnik Lab
%   UC Berkeley
%   20160815
%
%   20162024 EDIT: added parfor_progressbar. cleaned up unnecessary
%   commented out code
%
%   20180608 EDIT: added ability to process multiple directories



% Have user specify the video directory
% vidDir = uigetdir('F:\wtavi', 'Select the video directory to analyze');
vidDir = uigetdir2('F:\wtavi', 'Select the video directory to analyze');
% [~, dirName,~] = fileparts(vidDir);
% dirCell = {'FID734'};
% whiskerMat = [1];

% user input
% prompt   = {'Number of whiskers','Face position'};
% dlgTitle = 'Whisker Tracker Parameters';
% numLines = 1;
% %default  = {'-1','1650','680'};
% default  = {'-1','Face pos: top, bottom, left, right'}; % -1 means 'classify' will figure out the length threshold for classification
% usrInput = inputdlg(prompt,dlgTitle,numLines,default);
% 
% if isempty(usrInput)
%     disp('process canceled')
%     return
% end
% 
% whiskerMat = str2double(usrInput{1});
% faceX      = usrInput{2};

% default parameters (no user input)
whiskerMat = -1; % all whiskers
faceX      = 'bottom'; % face position
%faceY      = usrInput{3};

% fileMap  = [];
for session = 1:length(vidDir)
    [~, dirName,~] = fileparts(vidDir{session});
    mov_path = 'F:\wtavi\';
    mes_path = ['F:\tracking_data\' dirName filesep 'measure'];
    trc_path = ['F:\tracking_data\' dirName filesep 'trace'];

    fileMap  = [];
%     dirName = dirName{session};
%     num_whiskers = whiskerMat(session);
    num_whiskers = whiskerMat(1);
    
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
    hbar = parfor_progressbar(length(fstruct),[dirName ' Tracing whiskers...']);  %create the progress bar
    parfor k = 1:length(fstruct)
        hbar.iterate(1);   % update progress by one iteration
        fname = fstruct(k).name;
        system(['trace ' mov_path filesep dirName filesep fname ' ' trc_path filesep fname '.whiskers'])
    end
    close(hbar);
    %% Measure/Classify/Reclassify Trace Files
    hbar = parfor_progressbar(length(fstruct),[dirName ' Extracting measurements...']);
    fileCount   = 0;
    filesMissed = 0;
    for k = 1:length(fstruct)
        hbar.iterate(1);   % update progress by one iteration
        fname = fstruct(k).name;
        
        status = system(['measure --face ' faceX ' ' trc_path filesep fname '.whiskers' ' '...
            mes_path filesep fname '.measurements']);
        
        status = system(['classify ' mes_path filesep fname '.measurements' ' '...mov
            mes_path filesep fname '.measurements ' faceX ' ' '--px2mm 0.014 -n '...
            num2str(num_whiskers)]); % mm/px = 0.0198
        
        %--limit0.01:500']); % mm/px = 0.0198
        
        status = system(['reclassify ' mes_path filesep fname '.measurements' ' '...
            mes_path filesep fname '.measurements' ' -n ' num2str(num_whiskers)]);
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
    close(hbar);
    
    save(['F:\tracking_data/' dirName filesep 'fileMap.mat'],'fileMap','-v7.3')
    
end

b = toc;

disp(['total time ' num2str(b/(3600)) 'hours'])
clear all





