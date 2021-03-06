# A data-set of piercing needle through deformable objects for Deep Learning from Demonstrations

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Dataset Structure](#Dataset-Structure)
* [Models](#Models)
* [How to Run the Scripts](#How-to-Run-the-Scripts)
* [System Requirements](#System-Requirements)
## About The Project

Suturing is a frequently used operation in most of the surgeries and developing a robotic system for autonomous robotic suturing can be of significant assistance to surgery team in order to reduce the physical fatigue and cognitive load. Since tissues are deformable objects, needle insertion and tissue manipulation are complex control tasks. During inserting the circular needle (see Figure) into the deformable tissue, the needle tip pushes the tissue, hence, the desired and actual path differ and the actual exit-point, drifts away from the desired exit-point resulting in a less effective grip for the stitch or (in some cases) failure of the stitch- as the stitch cuts the tissue. In practice, surgeons utilise Arm 2  to manipulate the tissue and ensure the desired and actual exit-points are the same. Surgeons only use visual feedbacks to predict the needle exit point (performed by Arm 1 in the Fgure) and close the control loop by commanding Arm 2 to push/pool the tissue. 
<p align="center">
  <img width="200" src="images/needleInsertion_1(2).png">
  <img width="200" src="images/needleInsertion_2(1).png">
</p>

The second complexity of the control task belongs to the high dimensional sensory data (Visual sensor) which makes the problem very hard for conventional control theory/Learning from Demonstration algorithms. Formulating such tasks and implementing the corresponding automatic controller is very complex and time-consuming. Deep Robot Learnng from Demonstration can be effectively utilized to address such a control problem. We have collected a dataset for piercing needle into an artificial tissue with da Vinci Research Kit (DVRK) to develop deep LfD models for robot control.


## Dataset Structure

The setup includes DVRK and three pairs of stereo cameras with calibrated pose relative to robot arms' base frames. An artificial tissue made of polyethylene with homogenous texture and Young's modulus of 1.5 GPa is the flexible tissue under operation. The setup is shown in the following figure.
<p align="center">
  <img width="600" src="images/datasetsetup(2).png">
</p>

The dataset includes 60 folders including the corresponding data for 60 successful needle insertion trials executed by an expert. In each video, the desired needle tip exit point is specified by a red cross sign and robot data are being logged after this point appears on the screen. The operation is recorded by three pairs of stereo cameras (six cameras in total) and DVRK arms synchronized joint space and end-effector data are stored. The calibration parameters which include the relative pose of two cameras in each pair and pose of each camera relative to robots' base frames are included in the dataset as well. The overall structure of the dataset is illustrated by the following figure.

<p align="center">
  <img width="800" src="images/DatasetDiagramnew.png">
</p>

As such, each folder include six videos corresponding to six cameras and a .mat file which contains robot stored data. The overal structure of robot data can be stated as follow:

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

In the /Data directory the 60 folders named as 1-60 include the six videos and the .mat file robot data for each trial. For an easier access, we have included the .csv format of the data in the /Data/CSV files/ directory as well. The default data collection for the robot kinematic was to save 4*4 homogenous transformation matrices; we have also transformed the orientation representation into unit quaternions and included the corresponding data in /Data/quaternion Pose/ directory. 

The videos for a sample trial for the six cameras are shown in the following. Each column belongs to one stereo pair (top row left camera and bottom row right camera of each pair).


<p align="center">
  <img width="250" src="/images/P1_left.gif">
  <img width="250" src="/images/P2_left.gif">
  <img width="250" src="/images/P3_left.gif">
  <img width="250" src="/images/P1_right.gif">
  <img width="250" src="/images/P2_right.gif">
  <img width="250" src="/images/P3_right.gif">
</p>



## Models

We have developed deep models for control action generation for the robot arm which manipulates the tissue to guide the needle tip to exit from a desired specified point. As such, we have deployed state of the art CNN and RNN architectures as as feature extractor and next state predicotr respectively. The baseline methods achieved satisfactory performance based on the prediction error. The architecture of the developed models is shown in the following. 

<p align="center">
  <img width="700" src="images/System(2).png">
</p>

In these models we make use of d and f components of robot data which contain Arm2 end-effecotr position and orientation in time step t and t+1 respectively. Camera calibration data is also concatenated with CNN laten vector and robot state vectors. For CNN block we have deployed architectures including AlexNet, VGG19, MobileNet, and ResNet. For the recurrent neural netwrok block we have used Single RNN, LSTM and GRU. Other complex models can be developed to further improve the control action generation. 

## How to Run the Scripts

First run the imgdataset.py script in the /preprocessing directory to turn the videos into images with a frequency equal to the logged robot data for each trial.

    > run imgdataset.py 
    
    required file: Data/datasetSize.csv
    
For that datasetSize.csv is required which is available in /Data directory. This is due to the fact thta since for different trials, the number of saved cells are different (because trials are variable in length 7~10 second). After saving the images, different models scripts can be run from /Models directory.

    > run customized_cnn.py, ResNet.py, MovileNet.py ...
    
    required files: Data/trainImageName.csv, testImageName.csv, Arm2_CS_new.csv, Arm2_NS_new.csv

For these scripts 1- trainImageName.csv and testImageName.csv files (which include the image name of the train and test sets respectively) and 2- Arm2_CS_new.csv and Arm2_NS_new.csv (which include current and next step pose of Arm2) are required which can be found in /Data directory.

## System Requirements

The models' scripts can be run both with Trensroflow1 and 2 versions. The suggested Open-CV version for running preprocessing scripts is <= 4.1.2. Scikit-learn is required for preprocessing and standardization.

   * Tensorflow== 1 & 2
   * python-opencv <= 4.1.2
   * Scikit-learn
    
We have used Tensorflow-gpu 2.2 with a NVIDIA GeForce RTX2080 graphic card with 8GB memory with CUDA 11.0 for training on Ubuntu 18.04 and the models took maximum an hour to train. 
* Tensorflow-gpu 2.2
* NVIDIA GeForce RTX2080
* CUDA 11.0
* Ubuntu 18.04

For the RNN models at least 16GB RAM is required to load the input data for preprocessing. 


 <a href="https://arxiv.org/abs/2012.02458 ">Read the paper here! </a> 
 
 ### Citation

If you find our work useful for your research, please cite:
```
@article{Ghalamzan2021,
  title={Learning needle insertion from sample task executions},
  author={Amir Ghalamzan E.},
  journal={Submitted to IEEE/RSJ International Conference Intelligent Robotic System (IROS)},
  year={2021}
}
```

### License

This project Code is released under the Apache License 2.0 (refer to the LICENSE file for details).
