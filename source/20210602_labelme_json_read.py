# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:31:44 2021

@author: SURIMWANG
"""

#%%
import cv2
import numpy as np
import json
import os
os.chdir('D:/kidney/source')

#%%
#json 파일 읽기전에 원본을 불러와서 비교 준비하기
image = cv2.imread('../image/SCAN_01.jpg')

#labelme로 라벨링한 json 파일 읽기
with open('../image/SCAN_01.json', "r", encoding='UTF8') as st_json:
    st_python = json.load(st_json)
#출처: https://rfriend.tistory.com/474 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

#만들어진 json파일 분석하기    
st_python
#size : 7 
#1 flags :?
#2 imageData:?
#3 ImageHeight:이미지 행의 크기
#4 imagePath:이미지 이름
#5 imageWidth:이미지 렬의 크기
#6 shapes:라벨링 값
#7 version:리벨미의 버전

#%% st_python의 6번 분석하기
st_python['shapes'][0]
#{'label': '의', 
# 'points': [[57.57575757575757, 103.03030303030302], #사각형의 첫번째 포인트
#  [93.33333333333333, 137.57575757575756]], #사각형의 두번째 포인트
# 'group_id': None,
# 'shape_type': 'rectangle',
# 'flags': {}}
## 결론: 라벨미로 라벨링이 끝나면 파이썬에서 불러와 좌표에 맞게 이미지에 사각형을 그려주고 그 포인트 값을 한줄로 바꿔서 저장하면 
## MNIST의 데이터셋과 유사하게 저장할 수 있는데 다만! 이미지 마다 사이즈가 다르기 때문에 28*28처럼 748사이즈로 맞출 수 없다는 문제가 있다.
## 이럴 경우에는 어떻게 해결 할 수 있을까?

# 글자일 경우에는 MNIST처럼 같은 사이즈로 만든다 쳐도 동물과같은 비정형한 모형의 경우에는 라벨링을 한 데이터셋은 어떤 형태가 될까?
## 

#이미지에 사각형 그리기
label = st_python['shapes'][0]['label']
x1 = int(st_python['shapes'][0]['points'][0][0])
y1 = int(st_python['shapes'][0]['points'][0][1])

x2 = int(st_python['shapes'][0]['points'][1][0])
y2 = int(st_python['shapes'][0]['points'][1][1])
f_point = tuple([x1, y1])
l_point = tuple([x2, y2])

cv2.rectangle(image, f_point, l_point, (255,0,0), 2)

#이미지 보기
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/draw_rectangle.jpg', image) # 파일 경로 & 명, 파일

#%% 반복문으로 라벨미에서 그린 사각형 모드 표시하기
for i in range(len(st_python['shapes'])):
    label = st_python['shapes'][i]['label']
    # 첫좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    
    f_point = tuple([x1, y1])
    l_point = tuple([x2, y2])
    
    cv2.rectangle(image, f_point, l_point, (255,0,0), 2)

#이미지 보기
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/draw_rectangle.jpg', image) # 파일 경로 & 명, 

#%% 이미지 잘라서 데이터셋의 형태로 만들어 보기 : 
# 이미지에 사각형을 그리는게 중요한게 아니라 좌표를 이용해서 이미지 값을 불러오는게 중요하다.

image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)


x1 = int(st_python['shapes'][0]['points'][0][0]) #0,0 첫좌표의 행값
y1 = int(st_python['shapes'][0]['points'][0][1]) #0,0 첫좌표의 열값
# 마지막 좌표
x2 = int(st_python['shapes'][0]['points'][1][0])
y2 = int(st_python['shapes'][0]['points'][1][1])
    

cropped_image = image[y1: y2, x1: x2].copy()
    
    
#이미지 보기
cv2.imshow('cropped_image',cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/test_result/{}.jpg'.format(i), cropped_image) # 파일 경로 & 명, 

#%% 반복문의로 변경
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
for i in range(len(st_python['shapes'])):
    # 첫 좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    #크롭 이미지
    if x1 > x2:
        if y1 > y2:
            cropped_image = image[y2: y1, x2: x1].copy()    
        else:
            cropped_image = image[y1: y2, x2: x1].copy()    
    else:
        if y1 > y2:
            cropped_image = image[y2: y1, x1: x2].copy()    
        else:
            cropped_image = image[y1: y2, x1: x2].copy()    
    
    #이미지 저장하기
    cv2.imwrite('../image/test_result/{}.jpg'.format(i), cropped_image) # 파일 경로 & 명, 























#%% 테두리 입히기 
import math

im = cv2.imread('../image/test_result/24.jpg')
row, col = im.shape[:2]
bottom = im[row-2:row, 0:col]
mean = cv2.mean(bottom)[0]

col_bordersize = (55-col)/2
row_bordersize = (55-row)/2

border = cv2.copyMakeBorder(
    im,
    top = math.ceil(row_bordersize),
    bottom = math.floor(row_bordersize),
    left = math.ceil(col_bordersize),
    right = math.floor(col_bordersize),
    borderType = cv2.BORDER_CONSTANT,  
    value = [mean, mean, mean]
)

cv2.imwrite('../image/test_result/{}.jpg'.format('test.jpg'), border) # 파일 경로 & 명, 
#%% 테두리 입히기  함수로 만들기
import math
image = cv2.imread('../image/test_result/24.jpg')

def border_make(image):
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    
    col_bordersize = (55-col)/2
    row_bordersize = (55-row)/2
    
    border = cv2.copyMakeBorder(
        image,
        top = math.ceil(row_bordersize),
        bottom = math.floor(row_bordersize),
        left = math.ceil(col_bordersize),
        right = math.floor(col_bordersize),
        borderType = cv2.BORDER_ISOLATED, #BORDER_ISOLATED  BORDER_CONSTANT
        value = [mean, mean, mean]
    )
    return border
    
#cv2.imwrite('../image/test_result/{}.jpg'.format('test.jpg'), border) # 파일 경로 & 명, 
    
#%% 반복문의로 변경
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
for i in range(len(st_python['shapes'])):
    # 첫 좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    #크롭 이미지
    if x1 > x2:
        if y1 > y2:
            cropped_image = image[y2: y1, x2: x1].copy()    
        else:
            cropped_image = image[y1: y2, x2: x1].copy()    
    else:
        if y1 > y2:
            cropped_image = image[y2: y1, x1: x2].copy()    
        else:
            cropped_image = image[y1: y2, x1: x2].copy()    
            
    img = border_make(cropped_image)
    #이미지 저장하기
    cv2.imwrite('../image/test_border/{}.jpg'.format(i), img) # 파일 경로 & 명, 
    
    
    
    
    
    
    
    
#%% 이미지 임계값 처리 후 반복문으로 데이터 셋 만들기
# 임계값 넘으면 완전 블랙 아니면 0으로 이미지 재 처리하기 
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
dst = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret ,image = cv2.threshold(dst,127,255,0)
 #%%   
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
for i in range(len(st_python['shapes'])):
    # 첫 좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    #크롭 이미지
    if x1 > x2:
        if y1 > y2:
            cropped_image = image[y2: y1, x2: x1].copy()    
        else:
            cropped_image = image[y1: y2, x2: x1].copy()    
    else:
        if y1 > y2:
            cropped_image = image[y2: y1, x1: x2].copy()    
        else:
            cropped_image = image[y1: y2, x1: x2].copy()    
    #cropped_image = 255 - cropped_image
    img = border_make(cropped_image)
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret ,img = cv2.threshold(dst,127,255,0)

    #이미지 저장하기
    cv2.imwrite('../image/test_threshold_result/{}.jpg'.format(i), img) # 파일 경로 & 명, 

#%% MNIST 데이터셋 구조로 만들기
#MNIST 분석하기
train = pd.read_csv("../data/train.csv") #MNIST 데이터셋 (42000, 785) 라벨 포함 1 + 748

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)     
    
    
st_python['shapes'][0]['label'] #0만 바꿔주면 됨.

# 라벨을 리스트로 만들기
data = st_python['shapes'] #341
label_lst = []
for i in range(len(data)):
    label = st_python['shapes'][i]['label']
    label_lst.append(label)

#%% 플랫한 데이터를 담을 데이터 프레임 만들기 
pixel_lst = []
for i in range(3025):
    name = 'pixel_{}'.format(i)
    print(name)
    pixel_lst.append(name)
    
    
#데이터 프레임 컬럼 입력하기
df= pd.DataFrame(columns= columns_lst)
    
#데이터 프레임 데이터 입력 테스트하기
data = pd.Series(flat_img)
test =data.values.reshape((1,-1))

df_test = pd.DataFrame(columns= columns_lst, data = test)
df_test = df_test.append(df_test, ignore_index=True)


#%%데이터 플랫하게 만들기
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
for i in range(len(st_python['shapes'])):
    # 첫 좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    #크롭 이미지
    if x1 > x2:
        if y1 > y2:
            cropped_image = image[y2: y1, x2: x1].copy()    
        else:
            cropped_image = image[y1: y2, x2: x1].copy()    
    else:
        if y1 > y2:
            cropped_image = image[y2: y1, x1: x2].copy()    
        else:
            cropped_image = image[y1: y2, x1: x2].copy()    
    #cropped_image = 255 - cropped_image
    img = border_make(cropped_image)
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret ,img = cv2.threshold(dst,127,255,0)
# 이미지 데이터 플렛하게 만들기
    flat_img = img.flatten()
# 데이터 프레임에 플렛한 이미지 값 입력하기
    #이미지 플랫화
    flat_img = pd.Series(flat_img)
    #이미지 구조 변경
    flat_img = flat_img.values.reshape((1,-1))
    df2 = pd.DataFrame(columns= columns_lst, data= flat_img)
    df = df.append(df2, ignore_index=True)
    
df.to_csv('../data/dataset_test.csv', encoding ='utf-8')
    
#%% RGB > HSV 변환후 H,S,V 하나씩 살펴보기 

#json 파일 읽기전에 원본을 불러와서 비교 준비하기
image = cv2.imread('../image/SCAN_01.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# 이미지 분리
h,s,v = cv2.split(hsv)

#이미지 보기
cv2.imshow('h',v)
cv2.waitKey(0)
cv2.destroyAllWindows()

#결론 h, s는 아무런 데이터가 없고 v 즉, 명암만 있는 이미지이다.
#기존의 계획 h,s,v로 분리한 뒤 이미지에서 특정 영역의 불필요한 데이터가 있거나 
#디텍션하기 좋은 데이터가 있을경우 그것을 활용해 디텍션 성능을 높이는것을 목표로 잡았으나
#v만 존재하는 이미지 이기에 기존이미지와 hsv의 이미지는 다른 차이가 없는것으로 밝혀졌다.
    
    