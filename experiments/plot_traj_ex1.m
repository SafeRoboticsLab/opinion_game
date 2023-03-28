clear; close all; clc

% Loads data.
game_param = '11';
% xs = double(readNPY(strcat('two_car/two_car_', game_param, '_xs.npy')));
xs = double(readNPY(strcat('two_car/two_car_', game_param, '_xs_replan.npy')));

% xs = double(readNPY(strcat('two_car/two_car_L0_xs.npy')));
% xs = double(readNPY(strcat('two_car/two_car_L1L0_xs.npy')));

XR_in = xs(1:4, :);
XH_in = xs(5:8, :);

% Sets parameters.
option.keep_traj  = true;
option.is_fading  = false;
option.t_skip     = 15;
option.N_interp   = 1;
option.t_start    = 1;
option.t_end      = 110;
option.pause      = 0;
option.UI         = false;
option.fps = Inf;

% Image rotation to align the front with the world-frame x-axis 
option.rotation = 0;

% Image coordinates of the centre of the back wheels
option.centre = [348; 203];

% Real world length for scaling
option.length = 4.5;

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

% Font size
fs = 25;


%% Plot vehicles
f = figure('Color','white');
f.Position = [0 660 2000 665];
set(gca,'FontSize',fs)
if ~option.UI
  set(gca,'Visible','off')
end
hold on
daspect([1,1,1])
xlimSpan = [-5, 110];
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

    if mod(t-t_start, option.t_skip)~=0 %&& t~=t_end 
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

return

%% Plot demo opinions
close all
zs = double(readNPY(strcat('two_car/two_car_21_zs_replan.npy')));

% zs = double(readNPY(strcat('two_car/two_car_L0_zs.npy')));

fs = 30;    % Font size
ts = 0.2;
t_end = 130;
fposition = [0 200 700 200];

% % P1's opinion (softmax)
% sigma_z1 = softmax(zs(1:2, 1:t_end));
% f = figure('Color','white');
% f.Position = fposition;
% set(gca,'FontSize',fs)
% hold on
% plot((1:t_end)*ts, sigma_z1(1,:), 'LineWidth', 2)
% plot((1:t_end)*ts, sigma_z1(2,:), 'LineWidth', 2)
% xlabel('Time (s)', 'Interpreter','latex')
% ylabel('$\sigma(z^1)$', 'Interpreter','latex')
% leg = legend('$\sigma_1(z^1)$','$\sigma_2(z^1)$', 'Interpreter','latex');
% set(leg,'Box','off')
% ylim([-0.2, 1.2])
% 
% % P2's opinion (softmax)
% sigma_z2 = softmax(zs(3:4, 1:t_end));
% f = figure('Color','white');
% f.Position = fposition;
% set(gca,'FontSize',fs)
% hold on
% plot((1:t_end)*ts, sigma_z2(1,:), 'LineWidth', 2)
% plot((1:t_end)*ts, sigma_z2(2,:), 'LineWidth', 2)
% xlabel('Time (s)', 'Interpreter','latex')
% ylabel('$\sigma(z^2)$', 'Interpreter','latex')
% leg = legend('$\sigma_1(z^2)$','$\sigma_2(z^2)$', 'Interpreter','latex');
% set(leg,'Box','off')
% ylim([-0.2, 1.2])

% P1's opinion
z1 = zs(1:2, 1:t_end);
f = figure('Color','white');
f.Position = fposition;
set(gca,'FontSize',fs)
hold on
plot((1:t_end)*ts, z1(1,:), 'LineWidth', 2)
plot((1:t_end)*ts, z1(2,:), 'LineWidth', 2)
xlabel('Time (s)', 'Interpreter','latex')
ylabel('$z^1$', 'Interpreter','latex')
leg = legend('$z^1_1$','$z^1_2$', 'Interpreter','latex');
set(leg,'Box','off')
ylim([-7, 7])

% P2's opinion
z2 = zs(3:4, 1:t_end);
f = figure('Color','white');
f.Position = fposition;
set(gca,'FontSize',fs)
hold on
plot((1:t_end)*ts, z2(1,:), 'LineWidth', 2)
plot((1:t_end)*ts, z2(2,:), 'LineWidth', 2)
xlabel('Time (s)', 'Interpreter','latex')
ylabel('$z^2$', 'Interpreter','latex')
leg = legend('$z^2_1$','$z^2_2$', 'Interpreter','latex');
set(leg,'Box','off')
ylim([-7, 7])

% % P1's attention
% att1 = zs(5, 1:t_end);
% f = figure('Color','white');
% f.Position = fposition;
% set(gca,'FontSize',fs)
% hold on
% plot((1:t_end)*ts, att1, 'LineWidth', 2)
% xlabel('Time (s)', 'Interpreter','latex')
% ylabel('$\lambda^1$', 'Interpreter','latex')
% ylim([-0.5, 4])
% 
% % P2's attention
% att2 = zs(6, 1:t_end);
% f = figure('Color','white');
% f.Position = fposition;
% set(gca,'FontSize',fs)
% hold on
% plot((1:t_end)*ts, att2, 'LineWidth', 2)
% xlabel('Time (s)', 'Interpreter','latex')
% ylabel('$\lambda^2$', 'Interpreter','latex')
% ylim([-0.5, 4])


%% Plot Hs color map
Hs = double(readNPY(strcat('two_car/two_car_L1_Hs.npy')));

for time = 17
    figure('Color','white')
    imagesc(Hs(:,:,time))
    xticks([1,2,3,4])
    yticks([1,2,3,4])
    set(gca,'FontSize',fs)
    title(strcat('step=',num2str(time)), 'Interpreter','latex')
end
