import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

# We use only folders 1-20. This code creates a robot state csv for a case when one camera is used for testing and 5 for training

# save the dataset size of first 20 folders
path1 = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/DatasetSize.csv'
df1 = pd.read_csv(path1, header=None)
x = np.array(df1)
x = x[0]
testSize20 = x[0:20]
sum = np.sum(testSize20)
print(sum)


# T1 contains 5 copied labels for 5 training image and T1 one label for the test
path = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/RobotData/e&f CSV'
os.chdir(path)
datasetSize = int(sum)
T1 = np.zeros([5 * datasetSize, 7])
T2 = np.zeros([datasetSize, 7])
counter = 0

print(T1.shape)
print(T2.shape)

for i in range (1,21):
    fileName = 'Test' + str(i) + '_e.csv'
    df = pd.read_csv(fileName, header=None)
    rot = np.zeros([3,3])   # contains the rotation matrix and is built from first 9 elements of each row
    position = np.zeros([1,3])  # contains the position and is built from the last 3 elements of each row
    for j in range (0, df.shape[0]):
        rot[0][0:3] = df.T[j][0:3]
        rot[1][0:3] = df.T[j][3:6]
        rot[2][0:3] = df.T[j][6:9]
        position[0][0:3] = df.T[j][9:12]
        r = R.from_matrix(rot)
        quat = r.as_quat()  # transform from rotation matrix to quaternion
        quat = np.reshape(quat, (1,4))
        pose = np.concatenate((quat, position), axis=1)
        T1[5*counter:5*counter+5, :] = pose
        T2[counter:counter + 1, :] = pose
        counter = counter + 1

T = np.concatenate([T1, T2])
col = ['q0', 'q1', 'q2', 'q3', 'x', 'y', 'z']
dfNew = pd.DataFrame(T)
print(dfNew.shape)
newFileName = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/RobotData/Arm1_NS_cam6test.csv'
dfNew.to_csv(newFileName, header=col, index=None)