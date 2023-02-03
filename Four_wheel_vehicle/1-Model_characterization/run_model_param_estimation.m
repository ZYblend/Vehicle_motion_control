%% run model
clear all
clc

%% electic motor circuit parameters
% (model source: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling)
k     = 0.0027;  % motor torque constant (N.m/amp) = electromotive force constant (V/rad/sec)
R     = 0.47;    % electric resistance  (ohm)
Vm    = 12;      % maximum battery voltage (v)
J_eff = 1e-5;    % moment of inertia of the rotor (kg.m^2)
b =  1e-5;        % motor viscous friction constant (N.m.s)

offset_wm = 0;   % offset for motor rotation speed

Theta1 = (k^2/R + b)/J_eff;
Theta2 = k*Vm/(R*J_eff);


%% bicycle tire model paramteres
m     = 2.7;       % vehicle mass (kg)
rg    = 9.49;      % total drive ratio
Rw    = 0.024;     % wheel radius (m)
J     = 1.5;       % vehicle rotation inertia (unknown)
L     = 0.256;     % length of wheelbase (m)

Cw1 = 1000;
Cw2 = 1000;

% Theta3 = Cw1/J;
% Theta4 = Cw1/m;

Theta3 = Cw2/J;
Theta4 = Cw1/J;
Theta5 = Cw1/m;
Theta6 = Cw2/m;


%% run system
% command satuation (for safety)
% sat_pwm = 0.7;
% sat_steering = pi/6;


%% parameter estimation
sensor_data = readmatrix('data/mix2.csv');
sensor_data = sensor_data(2:end,:);
filter_data = readmatrix('data/sensor_data_mix2.csv');


% Inputs
time = [sensor_data(:,1), sensor_data(:,1)];
throttle_cmd = [sensor_data(:,1), sensor_data(:,2)];
steering_cmd = [sensor_data(:,1), sensor_data(:,3)];
% Outputs
current = [sensor_data(:,1), sensor_data(:,10)];
yaw_rate = [sensor_data(:,1), filter_data(:,1)];
rot_speed = [sensor_data(:,1), filter_data(:,2)];
longitudinal_velocity = [sensor_data(:,1), filter_data(:,3)];
% lateral_velocity = [sensor_data(:,1), filter_data(:,4)];
lateral_velocity = [sensor_data(:,1), 0.5 * tan(sensor_data(:,3)) .* filter_data(:,3)];

T_final = sensor_data(end-10,1);

