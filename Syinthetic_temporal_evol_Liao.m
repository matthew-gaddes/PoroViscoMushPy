%% Simulations of uplift episodes in volcanoes
% 9 Jan 2025. C.N.L

addpath('/home/camila/Documents/UK_project/Review_deformation_volcaneos/Temporal_series_from_grabit/Images/Liao_2021_poro_visco_elastic/code');
addpath('/home/camila/Documents/UK_project/Review_deformation_volcaneos/Temporal_series_from_grabit/Images/Liao_2021_poro_visco_elastic/data');

load('parameter_example');

%% First case: Uplift and subsidences:

timescale=1;% 0: relaxation characteristic timescale; 1: difusion characteristic timescale
t_Tr=15;% Number of times of Time of relaxation (Tr) to plot
t=linspace(0,t_Tr,300);% vector of time in the simulations
r_ratio=6; %
X0=[0.1 0.9 0.7 0.9 0.12 1];% Parameter values that works for uplift followed by subsidences episodes
TdTr=10*X0(6);

for delta=0.01:0.01:0.05;
    for i=0.1:0.2:1;
        tinjj=i*10; %Time injection duration normalized by time of diffusion (Td)
        [surface_z,surface_rho]=get_surface(0,5/3,X0(3),X0(4),r_ratio,X0(1),X0(2),KlMr,MmMr,delta,TdTr,tinjj,t,timescale);
        surface_z_dimens=surface_z*3000; %Unnormalizing surface displacements in meters
        days=24*3600;% in seconds
        Tr=100*days;% Relaxation time duration
        Td=TdTr*Tr;% Diffusion time duration
        plot(t*Td/days,surface_z_dimens,'LineWidth',2); %Unnormalized time and displacements
        xlabel('days'); ylabel('Uz[m]');
        hold on;
    end
end

% Second case: Only uplift signals

legends={};
a=0;
figure;
for delta=0.001:0.005:0.04;
    for i=0.01:0.05:1;
    a=a+1;
    TdTr=0.1*X0(6);
    tinjj=i*10;
    t_Tr=3;
    t=linspace(0,t_Tr,500);
    [surface_z,surface_rho]=get_surface(0,5/3,X0(3),X0(4),r_ratio,X0(1),X0(2),KlMr,MmMr,delta,TdTr,tinjj,t,timescale);
    surface_z_dimens=surface_z*3000;
    % Td allow to have an estimation of how long the episode will last in days
    Td=1000*days;%In seconds 
    Tr=1/(TdTr*(1/Td));%
    plot(t*Td/days,surface_z_dimens,'LineWidth',2);
    hold on;
    xlabel('t[days]');
    end
end

