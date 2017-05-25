function trackwhiskers_beast(vidDir, whiskerMat, varargin)
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
%   TODO: enable processing of multiple selected directories

% Default parameters that can be adjusted
promptDir = 'F:\wtavi\';       % default directory when prompting user to select a file
faceX = 'left';                % location of mouse's face
saveDir = 'F:\tracking_data\'; % directory to save output to

%% Parse input arguments
index = 1;
while index<=length(varargin)
    try
        switch varargin{index}
            case {'dir','Dir'}
                promptDir = varargin{index+1};
                index = index + 2;
            case {'face','Face'}
                faceX = varargin{index+1};
                index = index + 2;
            case {'saveDir','SaveDir'}
                saveDir = varargin{index+1};
                index = index + 2;
            otherwise
                warning('Argument ''%s'' not recognized',varargin{index});
                index = index + 1;
        end
    catch
        warning('Argument %d not recognized',index);
        index = index + 1;
    end
end

if ~exist('vidDir','var') || isempty(vidDir)
    vidDir = uigetdir(promptDir,'Select directory to analyze');
    if isnumeric(vidDir)
        return
    end
end
if ischar(vidDir)
    vidDir = {vidDir};
end
numDir = numel(vidDir);

if numel(whiskerMat)==1 && numDir>1
    whiskerMat = repmat(whiskerMat,numDir,1);
elseif numel(whiskerMat)~=numDir
    error('# of elements in second input must equal 1 or number of directories input');
end


%% Trace whiskers
fileMap  = [];
for session = 1:numDir
    
    % Determine files and prepare folders for saving
    dirName = flip(strtok(flip(vidDir{session}),'/')); % name of folder files exist in
    fstruct = dir(fullfile(vidDir{session},'*.avi'));  % discover AVI files
    fname = {fstruct(:).name};                         % pull out filenames
    numFiles = numel(fname);
    mes_path = fullfile(saveDir,dirName,'measure');    % determine measurements folder
    if exist(mes_path,'dir') ~= 7                      
        mkdir(mes_path)
    end
    trc_path = fullfile(saveDir,dirName,'trace');      % determine trace folder
    if exist(trc_path,'dir') ~= 7
        mkdir(trc_path)
    end
    
    tic
    
    %% Parallel Process
    hbar = parfor_progressbar(numFiles,'Tracing whiskers...');  % create the progress bar
    parfor k = 1:length(fname)
        system(['trace ' fullfile(vidDir,fname{k}) ' ' fullfile(trc_path,[fname{k},'.whiskers'])])
        hbar.iterate(1);   % update progress by one iteration
    end
    close(hbar);

    %% Measure/Classify/Reclassify Trace Files
    hbar = parfor_progressbar(numFiles,'Extracting measurements...');
    fileCount   = 0;
    filesMissed = 0;
    for k = 1:numFiles
        msrName = fullfile(mes_path,[fname{k},'.measurements']);

        status = system(['measure --face ' faceX ' ' fullfile(trc_path,[fname{k},'.whiskers']) ' '...
            msrName]);
        
        status = system(['classify ' msrName ' '...mov
            msrName ' ' faceX ' ' '--px2mm 0.014 -n '...
            num2str(whiskerMat{session})]); % mm/px = 0.0198
        
        %--limit0.01:500']); % mm/px = 0.0198
        
        status = system(['reclassify ' msrName ' '...
            msrName ' -n ' num2str(whiskerMat{session})]);

        
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
        
        hbar.iterate(1);   % update progress by one iteration
    end
    close(hbar);
    
    save(fullfile(saveDir,dirName,'fileMap.mat'),'fileMap','-v7.3')
    
end

disp(['total time ' num2str(toc/(3600)) 'hours'])
clear all





