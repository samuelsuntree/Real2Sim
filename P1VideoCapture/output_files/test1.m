% Load the data from x.csv and y.csv
x = csvread('F:\edge_consistency_v1\output_files\1\1\final\x.csv');
y = csvread('F:\edge_consistency_v1\output_files\1\1\final\y.csv');

% Get the number of time frames (columns in the CSV file)
num_frames = size(x, 2);

% Create a figure for the animation
figure;

% Loop through each time frame and plot the shape
for t = 1:num_frames
    % Clear the previous plot
    clf;
    
    % Plot the current shape
    plot(x(:, t), y(:, t), '-');
    
    % Maintain the aspect ratio of the plot
    axis equal;
    
    % Set plot limits based on the data range for a better view
    xlim([min(x(:)), max(x(:))]);
    ylim([min(y(:)), max(y(:))]);
    
    % Add title to indicate the current frame
    title(['Time Frame: ', num2str(t)]);
    
    % Pause to create an animation effect (adjust the pause duration as needed)
    pause(0.02);
end
