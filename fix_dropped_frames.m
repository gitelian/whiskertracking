


function ang_shifted = fix_dropped_frames(msr_dir, xml_dir, ang_notsmooth)
% exp_name, exp_dir, 
% get experiment name and appropriate directories
% xml_dir = 'F:\tracking_data\FID1326_avi\dropped_frames\';

% get number of movies
num_movies = length(msr_dir);
    
% matrix where the first column contains a 0 or 1 (end drop, begin drop
% respectively) for the entire experiment. Indices will be corrected
% from local values to glabal experiment values (i.e. frame index
% 10,000 on movie 2 is changed to index 40,000 for the entire
% experiment.
drop_matrix = [];

% shifts indices by the appropriate amount
% TODO: determine how many frames are in a movie programmatically
frames_per_movie = 30000;

% iterate through all movies and look for corresponding xml file with
% dropped frame information

%%% FOR LOOP BEGIN %%%
%%% FOR LOOP BEGIN %%%

for movie_index = 1:num_movies
    [~, movie_name, movie_extension] = fileparts(msr_dir(movie_index).name);
    [~, movie_name, ~] = fileparts(movie_name); % remove .avi from name
    
    % look for dropped frame file of the same name
    xml_ls = dir([xml_dir movie_name '*']);
    
    % determine if a dropped frame file exists
    if length(xml_ls) == 1
        % correct indices
        
        % read in xml file as MATLAB structure using function provided on the
        % mathworks website.
        xmlStruct = parseXML([xml_dir xml_ls.name]);
        
        % determine number of children nodes there are
        num_children = length(xmlStruct.Children);
        
        % create temporary drop indices matrix that will be concatenated to the
        % drop_indices matrix. first column 1 or 0 (begin drop, end drop
        % respectively)
        temp_drop_indices = nan(num_children, 2);
        
        % iterate through each child node looking for 'EVENTMARKER'
        for child_index = 1:num_children
            
            if strcmp(xmlStruct.Children(child_index).Name, 'EVENTMARKER')
                % determine how many attributes exist for an EVENT
                num_attributes = length(xmlStruct.Children(child_index).Attributes);
                
                % if attributes exist find drop begin/end indices
                if num_attributes ~= 0
                    for attr_index = 1:num_attributes
                        
                        % determine if it is a begin or end drop event and
                        % indicate that in the temp_drop_indices matrix (col 1)
                        if strcmp(xmlStruct.Children(child_index).Attributes(attr_index).Name, 'Name')
                            
                            % Begin drop event
                            if strcmp(xmlStruct.Children(child_index).Attributes(attr_index).Value, 'Begin Drop Event')
                                event_id = 1;
                                temp_drop_indices(child_index, 1) = 1;
                                
                                % End drop event
                            elseif strcmp(xmlStruct.Children(child_index).Attributes(attr_index).Value, 'End Drop Event')
                                temp_drop_indices(child_index, 1) = 0;
                            end
                            
                        end
                        
                        % get the index for the dropped frame and add it to the
                        % temp_drop indices matrix (col 2)
                        if strcmp(xmlStruct.Children(child_index).Attributes(attr_index).Name, 'FrameIndex')
                            temp_drop_indices(child_index, 2) = str2double(xmlStruct.Children(child_index).Attributes(attr_index).Value);
                        end
                        
                        
                    end
                end
            end
        end
        
        % temp_drop_indices matrix is now completely filled out for a given xml
        % file. now nans will be removed, indices properly shifted, and the
        % matrix will be concatenated with drop_matrix
        temp_drop_indices_nonan = temp_drop_indices(all(~isnan(temp_drop_indices),2),:); % for nan - rows
        temp_drop_indices_nonan(:, 2) = temp_drop_indices_nonan(:, 2) + movie_index*frames_per_movie;
        drop_matrix = [drop_matrix; temp_drop_indices_nonan];
    end
end

% verify drop_matrix alternates from drop begin to drop end
for k = 1:size(drop_matrix, 1)
    if mod(k, 2) == 1
        if drop_matrix(k, 1) ~= 1
            error('pattern does not alternate begin, drop, begin...end')
        end
    elseif mod(k, 2) == 0
        if drop_matrix(k, 1) ~= 0
            error('pattern does not alternate begin, drop, begin...end')
        end
    end
end
disp('drop_matrix looks good!')

num_frames = length(ang_notsmooth);
num_dropped_frames = 0;
for k = 1:2:length(drop_matrix)
    num_dropped_frames = num_dropped_frames + (drop_matrix(k+1, 2) - drop_matrix(k, 2));
end
total_frames = num_frames + num_dropped_frames;
ang_shifted = nan(total_frames, 1);

% add nans for each dropped frame and shift over good frames

last_good_frame = 1;
shift = 0;
for k = 1:2:length(drop_matrix)
    begin_drop      = drop_matrix(k, 2);
    end_drop        = drop_matrix(k+1, 2);
    num_good_frames = begin_drop - last_good_frame;
    
    ang_shifted(last_good_frame:(begin_drop-1)) = ang_notsmooth(shift + (1:num_good_frames));
    
    shift = shift + num_good_frames;
    last_good_frame = end_drop + 1;
end
ang_shifted(last_good_frame:(num_frames-shift)) = ang_notsmooth(shift:end);









