import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R



# This is the case we want to have camera 6 as test camera


# first we read the 6*12 calibration.csv file in which every row has one camera's calibration

name = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/calibration.csv'
calibration = pd.read_csv(name, header=None)
calibration = calibration.T
print("calibration data shape: {}".format(calibration.shape))


# The we copy the rows in an appropriate manner for camera6 as test camera
T1 = []
T2 = []

for i in range(1, 3497):
    T1.append(calibration[0])
    T1.append(calibration[1])
    T1.append(calibration[2])
    T1.append(calibration[3])
    T1.append(calibration[4])
    T2.append(calibration[5])

T = np.concatenate([T1, T2])
calibration = pd.DataFrame(T)

# Then we transform the data to quaternion

T1 = np.zeros([calibration.shape[0], 7])
rot = np.zeros([3, 3])
position = np.zeros([1, 3])
for i in range(0, calibration.shape[0]):
    rot[0][0:3] = calibration.T[i][0:3]
    rot[1][0:3] = calibration.T[i][3:6]
    rot[2][0:3] = calibration.T[i][6:9]
    position[0][0:3] = calibration.T[i][9:12]
    r = R.from_matrix(rot)
    quat = r.as_quat()
    quat = np.reshape(quat, (1, 4))
    pose = np.concatenate((quat, position), axis=1)
    T1[i, :] = pose


dfNew = pd.DataFrame(T1)
FileName = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/RobotData/CalibrationCam6Test.csv'
dfNew.to_csv(FileName, index=None)
