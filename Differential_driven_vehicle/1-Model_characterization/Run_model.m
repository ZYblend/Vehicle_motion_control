% Run_this_file_for parameter estimation
clear all
clc

%% parameter
% a1 = 5.29140375889488;
a1 = 2.94204835631313;
% a2 = 5.05605242551437;
% d  = 0.107542007874855;                        % distance betwen medium point of axis of wheels and center of mass(m)
a2 = 2.67256528206959;
d = 0.05;

%% sampling time
Ts = 0.01;

%% dataset
% load dataset
% theta = load('dataset/est_par_test2').theta;
% x     = load('dataset/est_par_test2').x;
% y     = load('dataset/est_par_test2').y;
% 
% v     = load('dataset/est_par_test2').v;
% w     = load('dataset/est_par_test2').w;

theta = load('dataset/data5').theta;
x     = load('dataset/data5').x;
y     = load('dataset/data5').y;
v     = load('dataset/data5').v;
w     = load('dataset/data5').w;

% v = ones(length(theta),1)*v;
% w = ones(length(theta),1)*w;

%% covert theta to coninuous signal
theta_new = zeros(size(theta));
theta_new(1) = theta(1);
iter = 0;
for idx = 2:length(theta)
    dif = theta(idx)-theta(idx-1);
    if dif <-pi
        iter = iter +1;
    end
    theta_new(idx) = ( pi+(iter-1)*2*pi )*(iter>0.5)+ ( theta(idx)+pi*(iter>0.5) );
end


%% form time serise input-ouputs
time_serise = linspace(0,round(Ts*size(theta,1)),size(theta,1)).';

theta_time = [time_serise, theta_new];
x_time =  [time_serise, x];
y_time =  [time_serise, y];
v_time =  [time_serise, v];
w_time =  [time_serise, w];

%% initial condition
z_initial = [theta(1); x(1); y(1)];


% save('dataset1.mat','x','y','theta','v','w','-v7.3');