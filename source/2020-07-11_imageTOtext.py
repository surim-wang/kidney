# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:11:18 2020

@author: SURIMWANG
"""

#%% 이미지 txt로 가져와서 한줄씩 들어온 데이터 전처리하기
import numpy as np
import pandas as pd
import os
os.chdir('D:/MNIST/source')
from PIL import Image
import pytesseract 


#test_kor= pytesseract.image_to_string(Image.open('../image/test_img3.png'), lang = 'kor')
#test_eng= pytesseract.image_to_string(Image.open('../image/test_img3.png'), lang = 'eng')

test_kor_eng= pytesseract.image_to_string(Image.open('../image/test_img3.png'), lang = 'kor+eng')
text = test_kor_eng.split('\n') #122
#text_raw = text

# 텝으로 띄워진 행을 제거하자(의미없는 행)
for num in reversed(range(len(text))):
    if text[num] =='' or text[num] == ' ':
        del text[num]        



# 우선 눈에 보이는 결측치를 수정하자(노가다작업의 일부) ocr을 인력이 극복할수 밖에 없는 문제다.
for num, line in enumerate(text):
    text[num]= text[num].replace("|","")
    text[num]= text[num].replace("\80","WBC")
    text[num]= text[num].replace("Het","Hct")
    text[num]= text[num].replace("LOL-C(Hl","LDL-C(계산")
    text[num]= text[num].replace(";",":")
    text[num] = text[num].replace("_","")
    text[num] = text[num].replace("컴사명","검사명")

for num, line in enumerate(text):
    text[num] = text[num].split(' ')
    

for line in range(len(text)):
    for atom in range(len(text[line])):
        #print(line, atom)
        print(text[line][atom])
        text[line][atom] = text[line][atom].replace("[","")
        #text[line][atom] = text[line][atom].split(':')

        
#이제 해야할하는건 행마다 빈칸 제거
for num_h in reversed(range(len(text))):
    for num_l in reversed(range(len(text[num_h]))):
        if text[num_h][num_l] == '':
            del text[num_h][num_l]
            
# 접수일 잡을수있는지 확인해보려했던 코드
for row in text:
    for column in range(len(row)):
        if '접수일:' in row[column]:
            print(row[column][4:])


#이제 행을 불러오면서 검사명이 포한된 행 다음줄부터 검사명이 나올때까지 첫번째가 그릇테이블에 검사명과 일치하면 
# 아 아니다 테이블로 만들자. 

#test_boxes = pytesseract.image_to_boxes(Image.open('../image/test_img3.png'), lang = 'eng')
plate_table = pd.read_csv("../2020-08-22_중복제거테이블.csv", encoding = 'euc-kr')
#%%
#test_table = plate_table
for result_id in text:
    for num, std_result_id in enumerate(plate_table['검사명']):
        print(result_id[0], std_result_id)
        if result_id[0] == std_result_id:
            #plate_table.loc[plate_table['검사명'] == 'Cast', '결과'] = 10
            plate_table.iloc[num,1] = result_id[1]
# 결과가 잘못 인식된것 나중에 한번에 사용자 & 본사 담당자가 수정해줘야함.


def is_nan(x):
    return (x is np.nan or x != x)


drop_num_lst= []
for num in range(len(plate_table)):
    x = plate_table.iloc[num,1]
    if is_nan(x) == True:
        print(num)
        drop_num_lst.append(num)


# ocr 테이블에 없는 데이터 제거하기
plate_table = plate_table.drop(drop_num_lst)
plate_table = plate_table.reset_index()
plate_table = plate_table.drop('index', axis = 1)

# 잘못인식된 데이터 맞춤 수정하기
plate_table.iloc[1,1] = 5.1
plate_table.iloc[3,1] = 45.6
plate_table.iloc[6,1] = 130
plate_table.iloc[8,1] = 5.1
plate_table.iloc[22,1] = 5.1


# 결과에 담겨있는 문자형 숫자를 실수로 변경
def func_float(table, row, col):
    x= float(table.iloc[row,col]) 
    table.iloc[row,col] = x

for row in range(len(plate_table)):
    func_float(plate_table, row ,1)
    

print(plate_table)

plate_table.to_csv("../result/ocr2csv.csv", index= False, encoding='utf-8-sig' )    
#%% cv2 그레이 스케일 성공

fname = "../image/이력서사진.jpg"
fname = "../image/내가디자인한신발5.PNG"
fname = "../image/IMG_8960.jpg"

fname = "test_img3.jpg"
fname = "이력서사진.jpg"
fname = "내가디자인한신발5.PNG"

import cv2
fname = "../image/test_img3.jpg"
original = cv2.imread(fname, cv2.IMREAD_COLOR)
gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
unchange = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

cv2.imshow('Gray', unchange)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

test_gray_kor= pytesseract.image_to_string(gray, lang = 'kor')
test_gray_eng= pytesseract.image_to_string(gray, lang = 'eng')

cv2.imwrite('../image/gray.jpg', gray)


#%% 테이블 ocr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os 
os.chdir('D:/MNIST/source')
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract


### first
#read your file
#file=r"../image/syImg.jpg" 
file=r"../image/test_img1.PNG"
img = cv2.imread(file,0)
img.shape
#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,100,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
#thresh,img_bin = cv2.threshold(img,110,255,cv2.THRESH_BINARY |cv2.THRESH_TOZERO)
#cv2.THRESH_TRUNC
#thresh,img_bin = cv2.threshold(img_bin,190,255,cv2.THRESH_BINARY |cv2.THRESH_TRUNC)


#img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#plt.imshow(img_opening,cmap='gray')
#plt.imshow(img_closing,cmap='gray')
#plt.imshow(img,cmap='gray')

#
#
#for r in range(1, 2219):
#    for c in range(1,1569):
#        print(r, c)
#        if int(img[r+1, c-1]) > 230:
#            img[r,c] = 255
#        elif int(img[r+1, c-1]) < 50:
#            img[r,c] = 0
#                       
#            
#img_bin1 = img_bin  
#for r in range(1, 2219):
#    for c in range(1,1569):
#        print(r, c)
#        if int(img_bin[r+1, c-1]) + int(img_bin[r+1, c]) + int(img_bin[r+1, c+1]) + int(img_bin[r, c-1])  + int(img_bin[r, c+1]) + int(img_bin[r-1, c-1]) + int(img_bin[r-1, c]) + int(img_bin[r-1, c+1]) > 1530:
#            img_bin[r,c] = 255
#            
#
#for r in range(215, 259):
#    for c in range(45,48):
#        print(r, c)
#        img_bin[r,c] = 255



#for r in range(490, 895):
#    for c in range(45,48):
#        print(r, c)
#        img_bin[r,c] = 255
        
        
#inverting the image 
img_bin = 255-img_bin
cv2.imwrite('../image/cv_inverted.png',img_bin)
#Plotting the image to see the output
plotting = plt.imshow(img_bin,cmap='gray')
plt.show()


### second find to detect rectangular boxes.

# Length(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image 
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

### vertical line catch
#Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
#image_1 = cv2.erode(img_bin, ver_kernel[0:2], iterations=3)

vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
#vertical_lines = cv2.dilate(image_1, ver_kernel[0:3], iterations=3)

cv2.imwrite("../image/vertical.jpg",vertical_lines)
#Plot the generated image
#plotting = plt.imshow(image_1,cmap='gray')
plotting = plt.imshow(image_1,cmap='gray')
plt.show()


### horizon line catch
#Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite("../image/horizontal.jpg",horizontal_lines)
#Plot the generated image
plotting = plt.imshow(image_2,cmap='gray')
plt.show()



### combine the horizontal and vertical lines
# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
#Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("../image/img_vh.jpg", img_vh)
bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
#Plotting the generated image
plotting = plt.imshow(bitnot,cmap='gray')
plt.show()



# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

#Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
#Get mean of heights
mean = np.mean(heights)



#Create list box to store all boxes in  
box = []
# Get position (x,y), width and height for every contour and show the contour on image
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        box.append([x,y,w,h])
plotting = plt.imshow(image,cmap="gray")
plt.show()




#Creating two lists to define row and column in which cell is located
row=[]
column=[]
j=0
#Sorting the boxes to their respective row and column
for i in range(len(box)):
    if(i==0):
        column.append(box[i])
        previous=box[i]
    else:
        if(box[i][1]<=previous[1]+mean/2):
            column.append(box[i])
            previous=box[i]
            if(i==len(box)-1):
                row.append(column)
        else:
            row.append(column)
            column=[]
            previous = box[i]
            column.append(box[i])
print(column)
print(row)



#calculating maximum number of cells
countcol = 0
for i in range(len(row)):
    countcol = len(row[i])
    if countcol > countcol:
        countcol = countcol
        
        
        
        
#Retrieving the center of each column
center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]
center=np.array(center)
center.sort()




#Regarding the distance to the columns center, the boxes are arranged in respective order
finalboxes = []
for i in range(len(row)):
    lis=[]
    for k in range(countcol):
        lis.append([])
    for j in range(len(row[i])):
        diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)
        lis[indexing].append(row[i][j])
    finalboxes.append(lis)
    
    
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  
  
#
#from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer=[]
for i in range(len(finalboxes)):
    for j in range(len(finalboxes[i])):
        inner=''
        if(len(finalboxes[i][j])==0):
            outer.append(' ')
        else:
            for k in range(len(finalboxes[i][j])):
                y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                finalimg = bitnot[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(finalimg,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel,iterations=1)
                erosion = cv2.erode(dilation, kernel,iterations=1)

                
                out = pytesseract.image_to_string(erosion)
                if(len(out)==0):
                    out = pytesseract.image_to_string(erosion, config='--psm 3')
                inner = inner +" "+ out
            outer.append(inner)




#Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row),countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")
#Converting it in a excel-file
#data.to_excel("../result/output1.xlsx")
#data.to_excel("../result/output1_point_del.xlsx")
data.to_excel("../image/output_gray.xlsx")

#%%
import pkg_resources
pkg_resources.working_set.by_key['pytesseract'].version
