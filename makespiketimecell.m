function [spikeDepth, spikeTimeCell, numContacts ] = makespiketimecell(dirName, fChanNum, SpikeData_dir, electrodeMat,...
    depthMat, contactMat, stimsequence, LASTMOVIE)

%Make spikeTime cell
%  Each row will correspond to a unit and each column > 1 will correspond
%  to a trial and contain the times a particular unit fired during a
%  specific trial

% Needed inputs:
% f_header (e.g. 'ID732')
% fChanNum (e.g. {'Ch1','Ch3','Ch5','Ch7','Ch9'})
% electrodeMat
% depthMat
% contactMat (from HSV analysis)
% spikes (structure from UMS3k)
% stimsequence
% LASTMOVIE

unit_count = 1;
spikeTimeCell = {};
spikeDepth = [];
numContacts = [];

for trode = 1:length(fChanNum)
    s_dir_struct = dir([SpikeData_dir filesep dirName '*' fChanNum{trode} '*.mat']); %find spikes file
    
    if  ~isempty(s_dir_struct)
        disp(['loading: ' s_dir_struct.name])
        load([SpikeData_dir filesep s_dir_struct.name])            % loads spikes file
        stimsequence = stimsequence(1:LASTMOVIE);                  % SINCE ONLY THE FIRST TRIALS SHOULD HAVE MOVIES
        uniqassigns = unique(spikes.assigns);                      % get all unique unit ids
        good_spike_ids = spikes.labels(spikes.labels(:,2) < 3);    % remove unit ids labeled as multi-unit or garbage
        uniqassigns = intersect(uniqassigns,good_spike_ids);       % sometimes UMS2K saves labels of units that do not exist, this takes that into consideration
        uniqtrials = unique(stimsequence);                         % get the condition number
        disp(['units found: ' num2str(length(uniqassigns))])
        
        for unit = 1:length(uniqassigns);                          % iterates through each sorted unit found above
            num_electrode_contacts  = size(spikes.waveforms,3);    % find how many electrode contacts where used to sort spikes
            unitId = uniqassigns(unit);                            % get unit ID number
            unit_ind = find(spikes.assigns == unitId);             % get indices of all spike times for the entire experiment for the given unit
            tt1_wave = mean(spikes.waveforms(unit_ind,:,1),1);     % mean spike waveform for first contact
            tt2_wave = mean(spikes.waveforms(unit_ind,:,2),1);     % mean spike waveform for second contact
            tt3_wave = mean(spikes.waveforms(unit_ind,:,3),1);     % mean spike waveform for third contact
            if num_electrode_contacts < 4                          % if 3 contacts were used for spike sorting
                tt4_wave = nan;                                    % assigne nan to the "4th" contact
            else
                tt4_wave = mean(spikes.waveforms(unit_ind,:,4),1); % mean spike waveform for fourth contact
            end
            [~, min_ind] = min([min(tt1_wave),min(tt2_wave),min(tt3_wave),min(tt4_wave)]); % find the index of the largest mean waveform
            depthTemp = depthMat(trode,min_ind);                   % use largest waveform index to assign it to a depth corresponding to the depth of the electrode it is closet to
            electrodeTemp = electrodeMat(trode,min_ind);           % use largest waveform index to get contact number it is closet to
            spikeDepth = [spikeDepth;[electrodeTemp,depthTemp]];   % vertically append electrode contact number and deth to spikeDepth matrix
            spikeTimeCell{unit_count,1} = [fChanNum{trode} '-' num2str(unitId)]; % add channel number (first part of spike file name) and contact number for each unit to spikeTimeCell
            
            for k = 1:length(stimsequence)                         % iterates through all trials
                
                spikeTrialInd      = find(spikes.trials == k);          % get all spike time indices for a given trial
                spikeTimesInd      = intersect(spikeTrialInd,unit_ind); % sub-select spike time indices for the given unit (intersection of all spike time indices for the unit with all spike time indices for the trial)
                spikeTimesPerTrial = spikes.spiketimes(spikeTimesInd);  % get spike times for individual trial
                spikeTimeCell{unit_count,k+1} = spikeTimesPerTrial;     % assign spike times for the given unit and trial to a cell in spikeTimeCell
                
                if trode == 1
                    contactInd      = contactMat(:,1,k);                % get indices of contact from HSV analysis for given trial
                    contactOnsetInd = find(diff(contactInd) == 1)+1;    % calculate the contact onset from contactInd (contactInd is 1 during contact and 0 otherwise)
                    numContacts(k,1) = length(contactOnsetInd);         % count the number of contacts and assign to numContacts for the given trial
                end
            end   % end iterating through all trials
            
            unit_count = unit_count + 1;
        end
    end
end

end % end function

