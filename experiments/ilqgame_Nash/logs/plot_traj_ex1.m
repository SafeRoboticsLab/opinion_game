clear; close all; clc

% Loads data.
xs = double(readNPY('corridor/example1_baseline_xs.npy'));
XR_in = xs(1:4, :);
XH_in = xs(5:8, :);

% Sets parameters.
option.keep_traj  = true;
option.is_fading  = true;
option.t_skip     = 1; %3
option.N_interp   = 1;
option.t_start    = 4;
option.t_end      = 20;
option.pause      = 0;
option.UI         = false;
option.fps = Inf;
option.rotation = 0;    % Image rotation to align the front with the 
                        % world-frame x-axis 
option_H = option;      % Image coordinates of the back wheels centre 
option.centre = [348; 103];
option_H.centre = [50; 50];
option.length = 0.3;    % Real world length for scaling
option_H.length = 0.4;

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
xlimSpan = [-2.5, 2.5];
ylimSpan = [-1.0, 1.0];
xlim(xlimSpan)
ylim(ylimSpan)

% Plots the road.
% rd_bd_min = -0.5;
% rd_bd_max = 0.5;
% fill([xlimSpan(1), xlimSpan(1), xlimSpan(2), xlimSpan(2)],...
%      [rd_bd_min, rd_bd_max, rd_bd_max, rd_bd_min],...
%      [191,191,191]/255);    % Road color
% plot(linspace(xlimSpan(1), xlimSpan(2), 2),...
%      linspace(rd_bd_min, rd_bd_min, 2),...
%      'k-','LineWidth',5)    % Road boundaries
% plot(linspace(xlimSpan(1), xlimSpan(2), 2),...
%      linspace(rd_bd_max, rd_bd_max, 2),...
%      'k-','LineWidth',5)    % Road boundaries

% Plots agent movements.
cnt = 1;
for t = t_start:t_end

    if mod(t-t_start, option.t_skip)~=0 && t~=t_end 
        continue
    end

    % Top-down view of the agents.
    xR_plt = XR(1:3, t);
    [option.image, ~, option.alpha] =...
        imread('corridor/car_figures/ego_car.png');
    option.alpha = option.alpha*alpha_vec(cnt);
    [~, hR] = plot_vehicle(xR_plt', 'model', option);
    xH_plt = XH(1:3, t);
    if mod(cnt,2) == 0
        [option_H.image, ~, option_H.alpha] =...
            imread('corridor/car_figures/human.png');
    else
        [option_H.image, ~, option_H.alpha] =...
            imread('corridor/car_figures/human_2.png');
    end
    if ~option.keep_traj && t~=t_end && t>t_start
        delete(hH)
    end
    option_H.alpha = option_H.alpha*alpha_vec(cnt);
    [~, hH] = plot_vehicle(xH_plt', 'model', option_H);

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

plot(2.0, 0.0, 'Marker', 'Pentagram', 'MarkerSize', 30,...
    'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r')    % Robot's target
plot(-2.0, 0.0, 'Marker', 'Pentagram', 'MarkerSize', 30,...
    'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g')    % Human's target

if ~option.keep_traj
    delete(hH)
end

