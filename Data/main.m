clear 
clc


for i=1:60
    clear Data
    %S= dir(fullfile(num2str(1),'/Data.mat'))
    S = fullfile(num2str(1),'/Data.mat')
    load (S)
    i
    for j=1:160
        a(i,j,:,:) = Data{i,j}{1};
        b(i,j,:,:) = Data{i,j}{2};
        c(i,j,:,:) = Data{i,j}{3};
        d(i,j,:,:) = Data{i,j}{4};
        e(i,j,:,:) = Data{i,j}{5};
        f(i,j,:,:) = Data{i,j}{6};
        g(i,j,:,:) = Data{i,j}{7};
        h(i,j,:,:) = Data{i,j}{8};
    end
end

% a (1x7): the kinematic data of arm 1
% b(1x7): the kinematic data of arm 2
% c (4x4): the end-effector posiiton of the robot 1 with respect to robot base 1 (forward kinematics 1)
% d (4x4): the end-effector posiiton of the robot 2 with respect to robot base 2 (forward kinematics 2)
% e (4x4): the next-step end-effector posiiton of the robot 1 with respect to robot base 1 (forward kinematics 1)
% f (4x4): the next-step end-effector posiiton of the robot 2 with respect to robot base 2 (forward kinematics 2)
% g (6x2): the 2D tracking target point on the image located in Cam1-Cam6 respectively.
% h (1x3): the computed 3D position of the target point with respect to the left camera in the middle camera pair (P2-left)
 