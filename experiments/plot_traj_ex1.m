clear; close all; clc

% Loads data.
xs = double(readNPY('two_car/two_car_22_xs.npy'));
XR_in = xs(1:4, :);
XH_in = xs(5:8, :);

% Sets parameters.
option.keep_traj  = true;
option.is_fading  = true;
option.t_skip     = 5;
option.N_interp   = 1;
option.t_start    = [];
option.t_end      = [];
option.pause      = 0;
option.UI         = false;
option.fps = Inf;

% Image rotation to align the front with the world-frame x-axis 
option.rotation = 0;

% Image coordinates of the centre of the back wheels
option.centre = [348; 203];

% Real world length for scaling
option.length = 6;

% Interpolates data.
if option.N_interp == 1
    XR = XR_in;
    XH = XH_in;
else
    XR = [];
    for i = 1:size(XR_in,1)
        XR = [XR; interp(XR_in(i,:), option.N_interp)];
    end

    XH = [];
    for i = 1:size(XH_in,1)
        XH = [XH; interp(XH_in(i,:), option.N_interp)];
    end
end

% Start and end time.
if ~isempty(option.t_start)
    t_start = option.t_start;
    if t_start ~= 1
        t_start = t_start * option.N_interp;
    end
else
    t_start = 1;
end

if ~isempty(option.t_end)
    t_end = option.t_end * option.N_interp;
else
    t_end = length(XR);
end

% Transparency settings.
if option.keep_traj && option.is_fading
    alpha_vec = linspace(0.1,1,t_end-t_start+option.t_skip);
%         alpha_vec = logspace(-1,0,t_end-t_start+option.t_skip);
else
    alpha_vec = linspace(1,1,length(XR));
end

% Figure setup.
fs = 25;
f = figure('Color','white');
f.Position = [0 660 2000 665];
set(gca,'FontSize',fs)
if ~option.UI
  set(gca,'Visible','off')
end
hold on
daspect([1,1,1])
xlimSpan = [-5, 100];
ylimSpan = [-6, 13];
xlim(xlimSpan)
ylim(ylimSpan)

% Plots the road.
road_start = xlimSpan(1);
road_end = xlimSpan(2);
rd_bd_min = -3.5;
rd_bd_max = 10.5;
rd_center = (rd_bd_min+rd_bd_max)/2;
grey_rgb = [150,150,150]/255;
white_rgb = [255,255,255]/255;

toll_start = 50;
toll_end = 80;
toll_center = (toll_start+toll_end)/2;

%   -> Road color
fill([road_start, road_start, road_end, road_end],...
     [ylimSpan(1), ylimSpan(2), ylimSpan(2), ylimSpan(1)], grey_rgb);
%   -> Road boundaries
plot(linspace(road_start, road_end, 2),...
    linspace(rd_bd_min, rd_bd_min, 2),...
    '-', 'Color', white_rgb, 'LineWidth', 3)
plot(linspace(road_start, road_end, 2),...
    linspace(rd_bd_max, rd_bd_max, 2),...
    '-', 'Color', white_rgb, 'LineWidth', 3)
%   -> Center line of the road
plot(linspace(road_start, toll_start, 2),...
    linspace(rd_center, rd_center, 2),...
    '--', 'Color', white_rgb, 'LineWidth', 5)
plot(linspace(toll_end, road_end, 2),...
    linspace(rd_center, rd_center, 2),...
    '--', 'Color', white_rgb, 'LineWidth', 5)

% Plots toll stations.
option_toll = option;
option_toll.length = 30;

% TOLL 1
option_toll_1 = option_toll;
option_toll_1.centre = [2000; 400];
[option_toll_1.image, ~, option_toll_1.alpha] =...
        imread('two_car/car_figures/toll_1.png');
plot_vehicle([toll_center; rd_bd_max; 0]', 'model', option_toll_1);

% TOLL 2
option_toll_2 = option_toll;
option_toll_2.centre = [2000; 390];
[option_toll_2.image, ~, option_toll_2.alpha] =...
        imread('two_car/car_figures/toll_2.png');
plot_vehicle([toll_center; 3.5; 0]', 'model', option_toll_2);

% TOLL 0
option_toll_0 = option_toll;
option_toll_0.centre = [2000; 180];
[option_toll_0.image, ~, option_toll_0.alpha] =...
        imread('two_car/car_figures/toll_0.png');
plot_vehicle([toll_center; rd_bd_min; 0]', 'model', option_toll_0);


% Plots agent movements.
cnt = 1;
for t = t_start:t_end

    if mod(t-t_start, option.t_skip)~=0 && t~=t_end 
        continue
    end

    % Top-down view of the agents.
    xR_plt = XR(1:3, t);
    [option.image, ~, option.alpha] =...
        imread('two_car/car_figures/car_robot_y.png');
    try
        option.alpha = option.alpha*alpha_vec(cnt);
    catch
        option.alpha = option.alpha*alpha_vec(end);
    end
    [~, hR] = plot_vehicle(xR_plt', 'model', option);
    xH_plt = XH(1:3, t);
    [option.image, ~, option.alpha] =...
        imread('two_car/car_figures/car_human.png');
    if ~option.keep_traj && t~=t_end && t>t_start
        delete(hH)
    end
    try
        option.alpha = option.alpha*alpha_vec(cnt);
    catch
        option.alpha = option.alpha*alpha_vec(end);
    end
    [~, hH] = plot_vehicle(xH_plt', 'model', option);

    % Axis and title.
    if option.UI
        xlabel('$p_x$','Interpreter','latex','FontSize',1.2*fs)
        ylabel('$p_y$','Interpreter','latex','FontSize',1.2*fs)
        title(['$t = $',...
            num2str((t-1)*planner.ts/option.N_interp,'%.1f')],...
            'Interpreter', 'latex', 'FontSize',1.2*fs)
    end

    % Pausing settings.
    if t == t_start
        pause(1.0)
    else
        pause(option.pause)
    end

    % Delete the last frame
    if  t~=t_end
        if ~option.keep_traj
            delete(hR)
        end
    end
    cnt = cnt + option.t_skip;
end

if ~option.keep_traj
    delete(hH)
end
