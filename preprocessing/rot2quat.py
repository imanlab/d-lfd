import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R



path = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/RobotData/e and f'
os.chdir(path)
datasetSize = 0
for i in range (1,61):
    fileName = 'Test' + str(i) + '_e.csv'
    df = pd.read_csv(fileName, header=None)
    datasetSize = datasetSize + df.shape[0]

print(datasetSize)


T = np.zeros([6 * datasetSize, 7])
counter = 0
for i in range (1,61):
    fileName = 'Test' + str(i) + '_e.csv'
    df = pd.read_csv(fileName, header=None)
    rot = np.zeros([3,3])
    position = np.zeros([1,3])
    for j in range (0, df.shape[0]):
        rot[0][0:3] = df.T[j][0:3]
        rot[1][0:3] = df.T[j][3:6]
        rot[2][0:3] = df.T[j][6:9]
        position[0][0:3] = df.T[j][9:12]
        r = R.from_matrix(rot)
        quat = r.as_quat()
        quat = np.reshape(quat, (1,4))
        pose = np.concatenate((quat, position), axis=1)
        T[6*counter:6*counter+6, :] = pose
        counter = counter + 1


dfNew = pd.DataFrame(T)
newFileName = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/RobotData/Arm1_NS_Pose_Labeled.csv'
dfNew.to_csv(newFileName, header=False, index=False)