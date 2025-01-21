% Load the data from spines_interpolated.csv and spines123.csv
data2 = csvread('1/final/spines_interpolated.csv');
data = csvread('1/final/spines.csv');

% Extract time, x, and y coordinates from spines_interpolated.csv
time2 = data2(:, 1);   % First column is the time for spines_interpolated.csv
coordinates2 = data2(:, 2:end);  % The rest are x, y pairs

% Extract time, x, and y coordinates from spines123.csv
time = data(:, 1);   % First column is the time for spines123.csv
coordinates = data(:, 2:end);  % The rest are x, y pairs

% Create a figure for the animation
figure;

% Set up the video writer to save the animation as an MP4 file
v = VideoWriter('spines_animation_aligned.mp4', 'MPEG-4');
v.FrameRate = 10;  % Adjust the frame rate as needed
open(v);

% Loop through each time frame in spines_interpolated.csv
for t = 1:length(time2)
    % Clear the previous plot
    clf;
    
    % Extract x and y coordinates for the current time frame from spines_interpolated.csv
    x2 = coordinates2(t, 1:2:end);
    y2 = coordinates2(t, 2:2:end);
    
    % Find the corresponding time index in spines123.csv
    time_idx = find(time == time2(t));
    
    if ~isempty(time_idx)
        % Extract x and y coordinates for the corresponding time frame from spines123.csv
        x = coordinates(time_idx, 1:2:end);
        y = coordinates(time_idx, 2:2:end);
        
        % Plot the shape from spines123.csv with a thick blue line
        plot(x, y, 'b-', 'LineWidth', 2, 'DisplayName', 'Spines123');
        hold on;
    end
    
    % Plot the shape from spines_interpolated.csv with a thick red line
    plot(x2, y2, 'r-', 'LineWidth', 2, 'DisplayName', 'Spines Interpolated');
    
    % Maintain the aspect ratio of the plot
    axis equal;
    
    % Set plot limits based on the data range for a better view
    xlim([min([coordinates(:); coordinates2(:)]), max([coordinates(:); coordinates2(:)])]);
    ylim([min([coordinates(:); coordinates2(:)]), max([coordinates(:); coordinates2(:)])]);
    
    % Add title to indicate the current time
    title(['Time: ', num2str(time2(t))]);
    
    % Add legend to distinguish the two plots
    legend('Location', 'Best');
    
    % Capture the plot as a frame in the video
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % Pause to create an animation effect (adjust the pause duration as needed)
    pause(0.1);
end

% Close the video writer
close(v);

% Close the figure
close(gcf);
