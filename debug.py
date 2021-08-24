import matplotlib.pyplot as plt
import numpy as np
import cv2


import pandas as pd
df = pd.read_csv('Train_DefectType_PrithviAI.csv')
defect = df.loc[df['defect_flag'] == 1]['images id'].values

# print(df.head())


# print(df.columns)




# # print(val1)
# # val1 = df.loc[, :].values[1:]
# print(val1)

# img_sz = (4096, 1000)


# img = cv2.imread('Images/000272.png')


# pt1 = (int((val1[0] - val1[2]/2) * 4096), int((val1[1] - val1[3]/2) * 1000))
# pt2 = (int((val1[0] + val1[2]/2) * 4096), int((val1[1] + val1[3]/2) * 1000))


# img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)

# plt.imshow(img[:, :, ::-1])
# plt.show()
