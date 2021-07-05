# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:19:20 2021

@author: SURIMWANG
"""

#%%
import cv2
import numpy as np
import os
os.chdir('D:/MNIST/source')

image = cv2.imread('../image/mountain.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 그레이 스케일은 rgb값을 각각 가중치를 줘서 나오는 결과 값이다.
result = np.zeros((image.shape[0], 256), dtype=np.uint8) 
# np.zeros()함수를 이용해 x행,y열의 구조를 만드는데 값은 0이 들어간다.
# dtype=np.uint8으로 정했는데 gray 스케일에서 가장 잘 사용하는 bit depth이다(정밀도) 2^8을 말하는것으로 하나의 비트를 256개의 색으로 입력할수있다.


#히스토그램 그리기
hist = cv2.calcHist([image], [0], None, [256], [0,256])
cv2.normalize(hist,hist,0,255, cv2.NORM_MINMAX)

for x, y in enumerate(hist):
    cv2.line(result, (x, image.shape[0]), (x, image.shape[0]-y), 255)

dst= np.hstack([image[:,:,0], result])
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
