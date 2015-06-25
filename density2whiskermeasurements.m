%DENSITY2WHISKERMEASUREMENTS Calculates the reflective whisker amplitude and
%   set-point from a 3-d array of whisker density measurements (created by
%   whiskers2histogram function) and saves them to a file in the current
%   directory. The sector is defined by the user.
%
%   To get correct measurements sector edges should be traced out in the
%   following way:
%   1) Define first edge as the most protracted region of the sector. Click on
%   a point nearest the whisker pad and then on a point at the end of the edge
%   2) Define the second edge as the least protracted region. The follicle
%   point should be defined first!!!
%
%   UC Berkeley
%   Adesnik Lab
%   G. Telian
%   20150624

[dat_filename, dat_path] = uigetfile('.den', 'Select whisker density file');
load([dat_path dat_filename], '-mat');

num_stimuli   = size(whisker_density, 3);
set_point     = zeros(length(num_stimuli), 1);
amp           = zeros(length(num_stimuli), 1);
sector_coords = cell(num_stimuli);

for k = 1:num_stimuli
    figure('position', [0, 0, size(whisker_density,2), size(whisker_density,1)])
    imagesc(1-whisker_density(:,:,k), [0.8 1]);
    colormap gray
    title({['Stimulus: ' num2str(k) ' ' 'Define the edges of the sector'],...
        '!!! The first point of an edge should be closest to the whisker pad !!!'})
    hold on

    x = zeros(4,1);
    y = zeros(4,1);
    for k = 1:4
        [xtemp, ytemp] = ginput(1);
        x(k) = xtemp;
        y(k) = ytemp;
        if k == 1
            plot(xtemp, ytemp, '-ro')
        elseif k == 2
            plot(x(1:2), y(1:2), '-ro')
        elseif k == 3
            plot(xtemp, ytemp, '-bo')
        elseif k ==4
            plot(x(3:4), y(3:4), '-bo')
        end
    end

    pause(1.5)
    hold off

    sector_coords{k} = [x, y];
    close all
    vec1 = [x(2) - x(1), -(y(2) - y(1))];
    vec2 = [x(4) - x(3), -(y(4) - y(3))];

    cos_theta = dot(vec1, vec2)/(norm(vec1)*norm(vec2));
    amp_temp = acos(cos_theta)*180/pi;
    amp(k) = amp_temp;
    % Evan YOU HAVE TO CHANGE THIS LINE TO CALCULATE THE CORRECT SET-POINT FOR
    % YOU SETUP
    max_angle = max([acos(dot(vec1,[1 0])/norm(vec1))*180/pi,...
                    acos(dot(vec2, [1 0])/norm(vec2))*180/pi]);
    set_point(k) = max_angle - amp_temp/2;

end

save([dat_filename(1:end-4) '-density-set-point.mat'], '-v7.3')
