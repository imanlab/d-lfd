# Deep Robot Learning from Demonstration (D-RLfD): A Dataset For Autonomous Robotic Suturing

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Dataset Structure](#Dataset-Structure)
* [Models](#Models)
## About The Project

Suturing is a frequently used operation in most of the surgeries and developing a robotic system for autonomous robotic suturing can be of significant assistance to surgery team in order to reduce the physical fatigue and cognitive load. Since tissues are deformable objects, needle insertion and tissue manipulation is a complex control task. The second complexity of the control task belongs to the high dimensional sensory data (Visual sensor) which makes the problem very hard for conventional control theory/Learning from Demonstration algorithms. We have collected a dataset for piercing needle into an artificial tissue with da Vinci Research Kit (DVRK) to develop deep LfD models for robot control.


## Dataset Structure

The setup includes DVRK and three pairs of stereo cameras with calibrated pose relative to robot arms' base frames. An artificial tissue made of polyethylene with homogenous texture and Young's modulus of 1.5 GPa is the flexible tissue under operation. The setup is shown in the following figure.
<p align="center">
  <img width="700" src="images/datasetsetup(2).png">
</p>

The dataset includes 60 folders including the corresponding data for 60 successful needle insertion trials executed by an expert. The operation is recorded by three pairs of stereo cameras (six cameras in total) and DVRK arms synchronized joint space and end-effector data are stored. The calibration parameters which include the relative pose of two cameras in each pair and pose of each camera relative to robots' base frames are included in the dataset as well. The overal structure of robot data can be stated as follow:

<p align="center">
  <img width="700" src="images/cell(4).png">
</p>

a: joint space kinematic data of Arm 1 (1×6)

b: joint space kinematic data of Arm 2 (1×6)

c: pose of Arm 1 w.r.t its base frame (4×4)

d: pose of Arm 2 w.r.t its base frame (4×4)

e: pose of Arm 1 w.r.t its base frame at t+1 (4×4)

f: pose ofArm 2 w.r.t its base frame at t+1 (4×4)

g: 2D tracking target point on the image captured by 6 cameras(6×2)

h: computed 3D position of the target point w.r.t Camera 3(1×3)


## Models

We have developed deep models for control action generation for the robot arm which manipulates the tissue to guide the needle tip to exit from a desired specified point. As such, we have deployed state of the art CNN and RNN architectures as as feature extractor and next state predicotr respectively. The baseline methods achieved satisfactory performance based on the prediction error. Other complex models can be developed to further improve the control action generation. 
