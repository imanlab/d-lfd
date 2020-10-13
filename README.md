# Deep Robot Learning from Demonstration (D-RLfD): A Dataset For Autonomous Robotic Suturing

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Dataset Structure](#Dataset-Structure)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [References](#references)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About The Project

Suturing is a frequently used operation in most of the surgeries and developing a robotic system for autonomous robotic suturing can be of significant assistant to surgery team in order to reduce the physical fatigue and cognitive load. Since tissues are deformable objects needle insertion and tissue manipulation is a complex control task. The second complexity of the control task belongs to the high dimensional sensory data (Visual sensor) which makes the problem very hard for conventional control theory/Learning from Demonstration algorithms. We have collected a dataset for piercing needle into an artificial tissue with da Vinci Research Kit (DVRK) to develop deep LfD models for robot control.


## Dataset Structure

The dataset includes 60 folders including the corresponding data for 60 successful needle insertion trials executed by an expert. The operation is recorded by three pairs of stereo cameras (six cameras in total) and DVRK arms joint space and end-effector data are stored. The calibration parameters which include the relative pose of two cameras in each pair and pose of each camera relative to robots base frames are included in the dataset as well. The overla structure of robot data can be stated as follow:


### Prerequisites

You may list down all the system requirements and dependencies here. 

```
```

### Installation

Instructions for cloning the repo and building the project files go here. 

```
```

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos would be great! You can also provide links to more resources. 

## Roadmap

This section can mention the milestones that have already been achieved and the proposed features/ miltestones, etc. 

## References 

The references used for your work (research papers, blogs, books,etc.) can be listed here. 

## Contact 

Provide a list of developers/maintainers email-ids of the repository so as to help resolve the issues future users may come across.


## Acknowledgements

If your work is an extension of an existing repository/research work, the original work can be mentioned here. 
This template is inspired from a great template made available by @othneildrew at https://github.com/othneildrew/Best-README-Template/blob/master/README.md. 
