




dirName  = 'FID755';%732;734;755
f_header = 'ID755';%732;734;755
tipDepth = 930;%920;954;930

electrodeMat = [8 20 3 21;...
    11 28 16 22;...
    4 29 10 30;...
    12 25 2 23;...
    5 17 9 24;...
    13 31 1 32;...
    15 18 7 26;...
    4 19 6 27];
depthMat = reshape(tipDepth - (0:25:775),4,8)'; % for 32 chan 25um spacing probe

mcdStrct = dir(['C:\Users\Greg\Documents\MATLAB\MCdatas\' '*' dirName(4:end)...
    '*.phy']);
load(['C:\Users\Greg\Documents\MATLAB\MCdatas\' mcdStrct.name],'MCdata','stimsequence','-mat')

contactStrct = dir(['C:\Users\Greg\Documents\MATLAB\TrackingData\ContactTimes\'...
    dirName '*.mat']);
load(['C:\Users\Greg\Documents\MATLAB\TrackingData\ContactTimes\' contactStrct.name]);

load(['C:\Users\Greg\Documents\MATLAB\TrackingData\ProcessedMeasurements\' dirName...
    '-data-cells.mat'])

LASTMOVIE     = size(angleCell,1);
% fChanNum      = {'Ch6','Ch1','Ch5','Ch7'};
fChanNum     = {'Ch01','Ch03','Ch05','Ch07','Ch09','Ch11','Ch13','Ch15'};
SpikeData_dir = 'C:\Users\Greg\Documents\MATLAB\SpikeData';

[run,notRun] = classify_run_trials(MCdata,30000,'674',200,25,1.250,3.0,1,1);

STIMSTART = 1.200;
STIMSTOP  = 2.200;

%% Plot angle for a single trial
% 
trialNum = 32;

contactMatTrial = double(contactMat(:,1,trialNum));
contactTimes = camTime(logical(contactMatTrial));
contactMatNan = contactMatTrial;
contactMatNan(contactMatNan == 0) = nan;

figure
subplot(2,1,1)
plot(camTime,angleCell{trialNum,1}(:,1),'k',camTime,contactMatNan*175)
ylim([90 180])
ylabel('deg')
xlabel('time (s)')
xlim([1.2 2.2])

subplot(2,1,2)
plot(camTime,phaseCell{trialNum,1}(:,1),'k',camTime,contactMatNan*-0.001);
xlim([1.2 2.2])

% subplot(2,1,2)
% plot(camTime,curveCell{trialNum,1}(:,1),'k',camTime,contactMatNan*-0.001);
% xlim([1.2 2.2])

%% Make spikeTime cell
[spikeDepth, spikeTimeCell, numContacts ] = makespiketimecell(f_header, fChanNum, SpikeData_dir, electrodeMat,...
    depthMat, contactMat, stimsequence, LASTMOVIE);

%% Make Contact PSTH and spike and time vectors/cells
uniqtrials           = unique(stimsequence);
num_uniqtrials       = length(uniqtrials);
BINSIZE              = 0.001;
censor_time          = 0.007;
bins                 = (censor_time:BINSIZE:1)'; % bins start at 7ms because spikes immediately proceeding a contact are ignored as they can be from the previous contact
contactPsth          = cell(size(spikeTimeCell,1),num_uniqtrials);
contactSpikesAndTime = cell(size(spikeTimeCell,1),num_uniqtrials);
spikesPerContactMean = zeros(size(spikeTimeCell,1),num_uniqtrials);
spikesPerContactStd  = zeros(size(spikeTimeCell,1),num_uniqtrials);
timeBetweenContacts  = zeros(size(spikeTimeCell,1),num_uniqtrials); 
stimTime             = zeros(size(spikeTimeCell,1),num_uniqtrials); 
contactsPerPosition  = cell(1,num_uniqtrials);
contactCurves        = cell(1,num_uniqtrials);
contactCurvesTime    = camTime - camTime(1);
CONTACTCURVELENGTH   = length(contactCurvesTime);

for unit = 1:size(spikeTimeCell,1)                                         % iterate through all units
    disp(['Getting data for unit: ' num2str(unit) '/' num2str(size(spikeTimeCell,1))])
    for pos = 1:num_uniqtrials                                             % iterate through all conditions
        
        countsTemp    = [];                                                % zero out countsTemp
        spikeRateTemp = [];                                                % zero out spikeRateTemp
        curveTemp     = nan(1,length(contactCurvesTime));                  % nan out curveTemp vector
        posInd        = find(stimsequence == pos);                         % find trial indices for the given condition (i.e. position)
        runPosInd     = intersect(posInd,run);                             % sub-select running trials for the given condition via intersecting condition indices with run indices
        
        if unit == 1
            contactTemp = [];                                              % initialize contactTemp to store the number of contacts
        end
        
        for k = 1:length(runPosInd)                                        % iterate through all running trials for the given condition
            contactInd   = contactMat(:,1,runPosInd(k));                   % get contact data for trial
            contactOnsetInd = find(diff(contactInd) == 1)+1;               % calculate contact onset indices for trial
            contactTimes = camTime(uint64(contactOnsetInd));               % get contact times for given trial trial (not sure why I convert type to unit64. probably got some indexing error where MATLAB didn't want to use a float to index...althoug now it seems to work 20141019)
            ind2remove   = find(contactTimes < STIMSTART | contactTimes > STIMSTOP); % get indices of contact times outside of the stimulus window (BUG 1: I WAS ONLY KEEPING CONTACT TIMES AFTER STIMSTOP!!)
            contactOnsetInd(ind2remove) = [];                              % remove contact onset indices outside of stimulus window
            contactTimes(ind2remove) = [];                                 % remove contact times outside of stimulus window
            
            if unit == 1                                                   % counts the number of contact per trial for each condition
                if isempty(contactTimes)                                   % BUG 2: THIS WAS SUBTRACTING 1 FROM ALL COUNTS
                    lengthContactTimes = 0;
                else
                    lengthContactTimes = length(contactTimes);
                end
                contactTemp = [contactTemp;lengthContactTimes];
            end
            
            if ~isempty(contactTimes) % check that contacts occurred for the trial
%                   contactTimes(end)-contactTimes(1)                                                         %%% MAKE CONTACT PSTH %%%
                for contact = 1:length(contactTimes)                       % iterate through all contacts for a trial
                    
                    if contact < length(contactTimes)                      % iterate up to the second to last contact
                        spikeTimesTemp = spikeTimeCell{unit,runPosInd(k)+1}; % get spike times for the current trial
                        spikeTimesTemp(spikeTimesTemp >= (contactTimes(contact+1)+censor_time) ) = []; % remove spike times that occur 7ms after the next contact
                        temp = histc(spikeTimesTemp,bins+contactTimes(contact)); % count the number of spike times starting 7ms after contact and ending 7ms after the next contact
                        countsTemp = [countsTemp;temp];                          % vertically concatenate counts to temporary count vector
                        spikeRateTemp = [spikeRateTemp;...
                            [sum(temp),(contactTimes(contact+1)-contactTimes(contact))]]; % count the number of spikes that occurred between contacts. the time between contacts is given by the second enty (i.e. the difference between contact times)
                        %%% Get curve segments for each contact
                        curve = curveCell{runPosInd(k),1}(contactOnsetInd(contact):contactOnsetInd(contact+1));
                        temp = nan(1,CONTACTCURVELENGTH);
                        if length(curve) > CONTACTCURVELENGTH
                            temp(1:CONTACTCURVELENGTH) = curve(1:CONTACTCURVELENGTH);
                        else
                            temp(1:length(curve)) = curve;
                        end
                        curveTemp = [curveTemp;temp];
                        %%% End get curve segments for each contact 
                    else
                        % I ignore the spikes assciated with the last
                        % contact in the stim period because it can be
                        % close to the end of the stimulus. The light off
                        % rebound can confound the spikes/contact
                        % measurements.
                    end
                end
            end
        end
        
        if unit == 1
            contactsPerPosition{1,pos} = contactTemp;
            contactCurves{1,pos}       = curveTemp;
        end
        
        numContactsTemp = size(countsTemp,1);
        if ~isempty(countsTemp) && (numContactsTemp > 3); % if there are more than 3 contacts
%             disp(' ')
%             disp(' ')
%             disp(['spike rate: ' num2str(sum(spikeRateTemp(:,1))/sum(spikeRateTemp(:,2)))])
%             disp(['pos: ' num2str(pos)])
            temp                           = sum(countsTemp,1)./(numContactsTemp); % count the number of spikes across all contacts for each bin *BINSIZE);%(binCount');
            temp(isnan(temp))              = 0;
            contactPsth{unit,pos}          = temp;                         % store spikes/contact to contactPsth
            contactSpikesAndTime{unit,pos} = spikeRateTemp;
            spikesPerContactMean(unit,pos) = mean(spikeRateTemp(:,1));
            spikesPerContactStd(unit,pos)  = std(spikeRateTemp(:,1));
            timeBetweenContacts(unit,pos)  = mean(spikeRateTemp(:,2));
%             disp(['times: ' num2str(sum(spikeRateTemp(:,2))/length(runPosInd))])
            stimTime(unit,pos)             = sum(spikeRateTemp(:,2))/length(runPosInd);
        else
            contactPsth{unit,pos}          = zeros(1,length(bins));        % assign zeros to contactPsth if there were no contacts
        end
        
    end
end
%%
n = 3;
figure;subplot(2,1,1);bar(bins,contactPsth{n,3});subplot(2,1,2);bar(bins,contactPsth{n,9});
%% Contact Rate Per Position


trialDur = STIMSTOP - STIMSTART;
contactRateMean = zeros(1,length(contactsPerPosition)-1);
contactRateStd  = zeros(1,length(contactsPerPosition)-1);
numSamp     = zeros(1,length(contactsPerPosition)-1);
normContact = {};

for k = 1:length(contactsPerPosition)
    contactRateMean(k) = mean(contactsPerPosition{1,k}/(STIMSTOP-STIMSTART));
    contactRateStd(k)  = std(contactsPerPosition{1,k}/(STIMSTOP-STIMSTART));%/sqrt(length(contactsPerPosition{1,k}));
    numSamp(k)     = length(contactsPerPosition{1,k});
    
    contactRateTemp = contactsPerPosition{1,k}/(STIMSTOP-STIMSTART);
end

ttest2(contactsPerPosition{1,1},contactsPerPosition{1,7})

figure
barwitherr(contactRateStd,contactRateMean)
xlabel('Position')
ylabel('Mean Contacts/Sec')

reshapeContactMean = zeros(3,6);
reshapeContactStd  = zeros(3,6);
reshapeContactMean(1,:) = contactRateMean(1:6);
reshapeContactMean(2,:) = contactRateMean(7:12);
% reshapeContactMean(3,:) = contactMean(13:18);%%%%%%%%%%
reshapeContactStd(1,:)  = contactRateStd(1:6);
reshapeContactStd(2,:)  = contactRateStd(7:12);
% reshapeContactStd(3,:)  = contactStd(13:18);%%%%%%%%%

figure
reshapeContactMean = reshape(contactRateMean,6,2);%%%%%%%
reshapeContactStd  = reshape(contactRateStd,6,2);%%%%%%%%%%
barweb(reshapeContactMean,reshapeContactStd,[],{'Pos1','Pos2','Pos3','Pos4','Pos5','Control'},...
    'Contact Rate Per Position','Positions','Contact Rate (cont/sec)')

[~,maxContactPos] = max(contactRateMean(1:6));

%% Spikes as a function of phase, make PSTH

spikePhase     = cell(size(spikeTimeCell,1),num_uniqtrials);
inds2remove    = find(camTime < STIMSTART | camTime > STIMSTOP);
phaseBins      = linspace(-pi,pi,25)';
binsize        = 0.01;
spikeBins      = 0:binsize:STIMSTOP;
PSTH           = cell(size(spikeTimeCell,1),num_uniqtrials);
stimInd        = find(spikeBins>=STIMSTART & spikeBins<=STIMSTOP);
baseInd        = 1:length(stimInd);
indRemoveSplit = find(diff(inds2remove)~=1)+1;
numWhiskCyclesMat = zeros(num_uniqtrials,1);


for unit = 1:size(spikeTimeCell,1)
    for pos  = 1:num_uniqtrials
        posInd    = find(stimsequence == pos);
        runPosInd = intersect(posInd,run);
        phaseTemp = [];
        setPointTemp = [];
        spikeTemp = [];
        numWhiskCycles = 0;
        for k = 1:length(runPosInd)
            
            %%% count the numbe of whisk cycles that occur during the stim period
            if unit == 1
                tempHighPass     = genButterFilter(angleCell{runPosInd(k),1}(:,1),1,50,4,'butter_acausal',fps);
                [~,protIndTemp]  = findpeaks(-tempHighPass);
                protIndTemp(protIndTemp < inds2remove(indRemoveSplit-1) |...
                    protIndTemp > inds2remove(indRemoveSplit))= [];
                numProtractions = length(protIndTemp);
                numWhiskCycles = numWhiskCycles + numProtractions;
                numWhiskCyclesMat(pos,1) = numWhiskCycles;
            end
            %%% End counting number of whisk cycles
            
            %%% Extract spike-phase values
            [counts,~] = histc(spikeTimeCell{unit,runPosInd(k)+1},camTime);% bin spike counts using the camTime vector so its easy to relate to HSV data. must add 1 because 1st column of spikeTimeCell is an electrode label
            counts(inds2remove) = 0;                                       % zero spike counts outside the stimulus period 
            phaseTemp = [phaseTemp; phaseCell{runPosInd(k),1}(logical(counts),1)]; % convert counts to logical index use this to 
            setPointTemp = [setPointTemp; setPointCell{runPosInd(k),1}(logical(counts),1)];
            %%% End extracting spike-phase values
            
            %%% Bin spikes for entire trial to make PSTH
            [spkCountTemp,~] = histc(spikeTimeCell{unit,runPosInd(k)+1},spikeBins);
            spikeTemp = [spikeTemp;spkCountTemp];
            %%% End binning spikes for entire trial
        end
        
        spikePhase{unit,pos} = phaseTemp;
        PSTH{unit,pos} = sum(spikeTemp,1)/(length(runPosInd));

    end
end

posWithContact = [maxContactPos,maxContactPos+6];

for unit = 1:size(spikeTimeCell,1)
    for pos = 1:4
        if pos == 1
%             figure;subplot(2,1,1);
%             bar(spikeBins,PSTH{unit,posWithContact(1)});
%             subplot(2,1,2)
%             bar(spikeBins,PSTH{unit,posWithContact(2)});
%             figure; barwitherr(spikesPerContactStd(unit,posWithContact),spikesPerContactMean(unit,posWithContact))
%             figure;plot(spikeBins,PSTH{unit,posWithContact(1)},'k',spikeBins,PSTH{unit,posWithContact(2)},'b')
        end
    end
end

% figure;plot(spikeDepth(:,2),spikesPerContactMean(:,posWithContact(1)),'*k',...
%     spikeDepth(:,2),spikesPerContactMean(:,posWithContact(2)),'*b')
% xlabel('unit depth (um)')
% ylabel('spikes/contact')
% title('Spikes per contact for sensory driven units at position with most contacts')
% legend('No Light','L4 Silencing','Location','NorthWest')
% 
% save([dirName '_spikesPerContact.mat'],'spikeTimeCell','spikeDepth','totalContactsPerPos',...
%     'spikesPerContactMean','posWithContact','spikesPerContactStd','PSTH','-mat');

%%% Calculate circular statistics on phase information (i.e. phase values
%%% every time unit spiked for every condition). This will allow us to
%%% identify phase tuned neurons and describe their modulation depth.
spikePhaseBin   = cell(size(spikePhase));
vectorStrength  = nan(size(spikePhase));
vectorDirection = nan(size(spikePhase));
vectorPvalue    = nan(size(spikePhase));
vectorZscore    = nan(size(spikePhase));

for unit = 1:size(spikePhase,1)
    for pos = 1:size(spikePhase,2)
        spikePhaseVals = spikePhase{unit,pos};
        if ~isempty(spikePhaseVals)
            temp = histc(spikePhaseVals,phaseBins);
            spikePhaseBin{unit,pos} = temp./numWhiskCyclesMat(pos,1); % units: spikes/cycle
            vectorStrength(unit,pos)  = circ_r(spikePhaseVals);
            vectorDirection(unit,pos) = circ_mean(spikePhaseVals);
            [p,z]                     = circ_rtest(spikePhaseVals);
            vectorPvalue(unit,pos)    = p; % tried cdRtest, value is same order of magnitude but had an average difference from circ_rtest of 0.0022 for data in FID732
            vectorZscore(unit,pos)    = z;
        end
    end
end
%%
maxContactPos = ones(size(spikeDepth,1))*maxContactPos;

save([dirName '_spikesAndBehavior.mat'],'spikePhaseBin','spikePhase','vectorStrength',...
    'vectorDirection','vectorPvalue','vectorZscore','phaseBins','numWhiskCyclesMat',...
    'spikeDepth','spikeTimeCell','stimsequence','run','bins','contactPsth','contactsPerPosition',...
    'spikesPerContactMean','posWithContact','spikesPerContactStd','PSTH',...
    'reshapeContactMean','reshapeContactStd','contactSpikesAndTime','-mat')
%%% End circular statistics on phase information

%% Make Plots
% 
% Contact triggered PSTHs

% for unit = 1:size(contactPsth,1)
%     maxVal = 0;
%     for k = 1:size(contactPsth,2)
%         
%         temp = max(contactPsth{unit,k});
%         if temp > maxVal
%             maxVal = temp;
%         end
%     end
%     if maxVal > 0  
%         figure;
%         for pos = 1:2
%             subplot(2,1,pos)
%             bar(bins,contactPsth{unit,posWithContact(pos)},'histc')
%             ylim([0 maxVal])
%             xlim([0 bins(end)])
%             if pos == 1
%                 ylabel('spikes/contact*bin')
%                 xlabel('time from contact (s)')
%                 title(['unit: ' num2str(unit)])
%             end
%         end
%     end
% end
% 
% %%% Plot spike/phase histograms
% posOfInterest = [maxContactPos,6,maxContactPos+6,12];
% tempBins = linspace(-pi,pi,25);
% for unit = 1:size(spikeTimeCell,1)
%     figure;
%     for pos = 1:4
%         subplot(2,2,pos)
%         counts = histc(spikePhase{unit,posOfInterest(pos)},tempBins);
%         bar(tempBins,counts./numWhiskCyclesMat(1,pos),'histc');ylabel('Spikes/Cycle')
%         ylim([0 0.05])
%         if pos == 1
%             title(['unit: ' num2str(unit)])
%         end
% %         rose(spikePhase{unit,posOfInterest(pos)},tempBins+pi)
%     end
% end

% %% Plot curves from two positions
% figure
% subplot(1,2,1)
% plot(contactCurvesTime,contactCurves{1,3}','b');hold on;
% plot(contactCurvesTime,nanmean(contactCurves{1,3}),'k','linewidth',3.0);hold off;
% ylim([minVal maxVal])
% xlim([0 bins(end)])
% subplot(1,2,2)
% plot(contactCurvesTime,contactCurves{1,4}','b');hold on;
% plot(contactCurvesTime,nanmean(contactCurves{1,4}),'k','linewidth',3.0);hold off;
% ylim([minVal maxVal])
% xlim([0 bins(end)])
% 
% % Spikes per contact
% unit = 1;
% for  unit = 1:size(contactPsth,1)
%     [maxYval,maxYvalInd] = max(spikesPerContactMean(unit,:));
%     if maxYval > 0
%         maxYval = maxYval + spikesPerContactStd(unit,maxYvalInd);
%         maxYval = maxYval + 0.05*maxYval;
%         figure
%         barwitherr(spikesPerContactStd(unit,:),spikesPerContactMean(unit,:))
%         ylim([0 maxYval])
%         xlabel('Position')
%         ylabel('Spikes/Contact')
%     end
% end
% 
% 








