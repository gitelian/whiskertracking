
fid = '0964'
fstruct = dir(['~/Documents/AdesnikLab/HSV_Data/' fid '*.phy'])
load(['~/Documents/AdesnikLab/HSV_Data/' fstruct.name], '-mat')
fstruct = dir(['~/Documents/AdesnikLab/Processed_HSV/*' fid '*.mat'])
load(['~/Documents/AdesnikLab/Processed_HSV/' fstruct.name] , '-mat')
[trialsran, trialswalk, runspeed, time] = classify_run_trials(MCdata, 30000,...
fid, 250, 25, 0.25, 2.0, 1);

num_trial_types = length(unique(stimsequence));

setpoint = cell(num_trial_types, 1);
amp      = cell(num_trial_types, 1);
mean_setpoint = zeros(num_trial_types, 1);
sem_setpoint  = zeros(num_trial_types, 1);
mean_amp = zeros(num_trial_types, 1);
sem_amp  = zeros(num_trial_types, 1);

for k = 1:num_trial_types
    run_inds = intersect(trialsran, find(stimsequence == k));
    temp_setpoint = [];
    temp_amp = [];
    temp_speed = [];
    for r = 1:length(run_inds)
        % Get good indices (i.e. ignore censor indices)
        good_inds = 1:length(camTime);
        censorCell{r}(censorCell{r} <= 0) = []; % Remove in future release
        censorCell{r}(censorCell{r} > length(camTime)) = []; % Remove in future release
        good_inds(censorCell{r}) = []; % Removes indices associated with bad frames
        temp_setpoint = [temp_setpoint; setPointCell{run_inds(r)}(good_inds)];
        temp_amp = [temp_amp; ampCell{run_inds(r)}(good_inds)];

        % Iterate through all good indices and record setpoint, amplitude and
        % the runspeed of the animal.
        speed = runspeed{run_inds(r)};
        t = time{run_inds(r)};
        for gind = 1:length(good_inds)
            [~, speed_ind] = min(abs(t - camTime(good_inds(gind))));
            temp_speed = [temp_speed;...
            [speed(speed_ind), setPointCell{run_inds(r)}(good_inds(gind)), ampCell{run_inds(r)}(good_inds(gind))]];
        end
    end
    %disp(any(isnan(temp_setpoint)))
    setpoint{k} = temp_setpoint;
    mean_setpoint(k) = nanmean(temp_setpoint);
    sem_setpoint(k) = nanstd(temp_setpoint)/sqrt(length(run_inds));

    amp{k} = temp_amp;
    mean_amp(k) = nanmean(temp_amp);
    sem_amp(k) = nanstd(temp_amp)/sqrt(length(run_inds));

    % Record runspeed data for each trial type
    speed_and_whisks{k} = temp_speed;
end

trialsran = 1:length(stimsequence);
speed_and_whisks = cell(num_trial_types);
camTimeBins = camTime(1):0.100:camTime(end);
runTime  = time{1};
for k = 1:num_trial_types
    run_inds = intersect(trialsran, find(stimsequence == k));
    temp_speed = [];
    for r = 1:length(run_inds)
        % Get good indices (i.e. ignore censor indices)
        good_inds = 1:length(camTime);
        censorCell{r}(censorCell{r} <= 0) = []; % Remove in future release
        censorCell{r}(censorCell{r} > length(camTime)) = []; % Remove in future release
        good_inds(censorCell{r}) = []; % Removes indices associated with bad frames

        % Iterate through all good indices and record setpoint, amplitude and
        % the runspeed of the animal.
        runSpeed = runspeed{run_inds(r)}*(2*pi*6/360);
%        t = time{run_inds(r)};
%        for gind = 1:length(good_inds)
%            [~, speed_ind] = min(abs(t - camTime(good_inds(gind))));
%            temp_speed = [temp_speed;...
%            [speed(speed_ind), setPointCell{run_inds(r)}(good_inds(gind)), ampCell{run_inds(r)}(good_inds(gind))]];
%        end
        %camTimeBins = STIMSTART:0.250:STIMSTOP;
        for binInd = 1:length(camTimeBins)-1
            setInd = find(camTime >= camTimeBins(binInd) & camTime < camTimeBins(binInd+1));
            setInd = intersect(setInd, good_inds);
            if ~isempty(setInd)
                runInd = find(runTime >= camTimeBins(binInd) & runTime < camTimeBins(binInd+1));
                runTemp = mean(runSpeed(runInd));
                runSetTemp = mean(setPointCell{run_inds(r)}(setInd));
                runAmpTemp = mean(ampCell{run_inds(r)}(setInd));
                temp_speed = [temp_speed;...
                [runTemp, runSetTemp, runAmpTemp]];
            end
        end
    end
    % Record runspeed data for each trial type
    speed_and_whisks{k} = temp_speed;
end


figure;
barweb(reshape(mean_setpoint, 6, 2), reshape(sem_setpoint, 6, 2));
ylim([90 160])
xlabel('Positions')
ylabel('Degrees')
title('Mean Set-Point')

figure;
barweb(reshape(mean_amp, 6, 2), reshape(sem_amp, 6, 2));
xlabel('Positions')
ylabel('Degrees')
title('Mean Amplitude')
ylim([0 20])

figure;
subplot(2,1,1)
scatter(speed_and_whisks{1}(:,1), speed_and_whisks{1}(:,2), 'ko')
xlim([0 100])
ylim([90 180])
xlabel('Speed cm/sec')
ylabel('Degrees')
title('Set-Point vs Runspeed')

subplot(2,1,2)
scatter(speed_and_whisks{7}(:,1), speed_and_whisks{7}(:,2), 'ko')
xlim([0 100])
ylim([90 180])
xlabel('Speed cm/sec')
ylabel('Degrees')
title('Set-Point vs Runspeed')
