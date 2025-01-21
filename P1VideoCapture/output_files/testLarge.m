% Define the number of rows and columns for the subplots
num_rows = 3;
num_cols = 3;

% Define the file paths and names
base_path = 'C:\Users\10521\Documents\GitHub\Aquaman\output\';
num_files = 9;

% Create a figure for the animation
figure;

% Set the figure size to 2K resolution
set(gcf, 'Position', [100, 100, 2560, 1440]);  % [left, bottom, width, height]

% Set up the video writer to save the animation as an MP4 file
v = VideoWriter('animation_grid_2k.mp4', 'MPEG-4');
v.FrameRate = 30;  % Adjust the frame rate as needed
open(v);

% Load all data into cell arrays for x and y
x_data = cell(1, num_files);
y_data = cell(1, num_files);
max_num_frames = 0;

% Variables to store global min and max values for axis limits
global_x_min = Inf;
global_x_max = -Inf;
global_y_min = Inf;
global_y_max = -Inf;

for i = 1:num_files
    x_file = fullfile(base_path, ['x_', num2str(i), '.csv']);
    y_file = fullfile(base_path, ['y_', num2str(i), '.csv']);
    x_data{i} = csvread(x_file);
    y_data{i} = csvread(y_file);
    
    % Update the maximum number of frames
    num_frames = size(x_data{i}, 2);
    if num_frames > max_num_frames
        max_num_frames = num_frames;
    end
    
    % Update global min and max for axis limits
    global_x_min = min(global_x_min, min(x_data{i}(:)));
    global_x_max = max(global_x_max, max(x_data{i}(:)));
    global_y_min = min(global_y_min, min(y_data{i}(:)));
    global_y_max = max(global_y_max, max(y_data{i}(:)));
end

% Loop through each time frame
for t = 1:max_num_frames
    % Clear the previous plots
    clf;
    
    % Loop through each subplot
    for i = 1:num_files
        % Select the subplot location
        subplot(num_rows, num_cols, i);
        
        % Determine the number of frames in the current dataset
        num_frames = size(x_data{i}, 2);
        
        % If the current frame exceeds the number of frames in this dataset,
        % hold the last frame
        if t <= num_frames
            plot(x_data{i}(:, t), y_data{i}(:, t), '-');
        else
            plot(x_data{i}(:, num_frames), y_data{i}(:, num_frames), '-');
        end
        
        % Maintain the aspect ratio of the plot
        axis equal;
        
        % Set global plot limits for better alignment
        xlim([global_x_min, global_x_max]);
        ylim([global_y_min, global_y_max]);
        
        % Add title to indicate the file number and current frame
        title(['File ', num2str(i), ' - Frame: ', num2str(t)]);
    end
    
    % Capture the frame for the video
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % Pause to control the animation speed
    pause(0.005);
end

% Close the video writer
close(v);

% Close the figure
close(gcf);
