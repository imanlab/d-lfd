import numpy as np
import pandas as pd
import os
from PIL import Image

# This is the case when we want to have camera6 ('camera_p3_right_data') as test camera

fname = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/DatasetSize.csv'
testSize = pd.read_csv(fname, header=None)
testSize = np.array(testSize)
subset = testSize[0, 0:20]
sum = np.sum(subset)
print(int(sum))
print(int(5*sum))
print(int(6*sum))


subFolder = 'camera_p1_left_data', 'camera_p1_right_data', 'camera_p2_left_data', 'camera_p2_right_data', 'camera_p3_left_data'
path = '/home/kiyanoush/UoLincoln/Projects/DeepIL Codes/Data'
os.chdir(path)
x=[]

for i in range(1,6):
    for j in range(1, int(testSize[0, i-1]) + 1):
        for f in subFolder:
            os.chdir(path+'/'+ str(i) + '/' + f)
            imgName = str(j) + '.jpg'
            image = Image.open(imgName)
            new_image = image.resize((224,224))
            new_image = np.array(new_image)
            x.append(new_image)


sf = 'camera_p3_right_data'
for i in range(1,6):
    for j in range(1, int(testSize[0, i-1]) + 1):
        os.chdir(path+'/'+ str(i) + '/' + sf)
        imgName = str(j) + '.jpg'
        image = Image.open(imgName)
        new_image = image.resize((224,224))
        new_image = np.array(new_image)
        x.append(new_image)

x= np.array(x)
print(x.shape)

savePath = '/home/kiyanoushs/Kiyanoush Codes/Needle Insertion/Data/Test1-20Cam6Test.npy'
np.save(savePath, x)
