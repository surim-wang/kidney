# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:31:14 2021

@author: SURIMWANG
"""

#%%
import cv2
import pandas as pd 
import numpy as np
import os
os.chdir('D:/MNIST/source')
from datetime import datetime, timedelta
from glob import glob

# 개별 데이터 to 종합 데이터로 
data_list = glob('D:/05.Kidney/data/개별데이터/*.xlsx')
col = pd.read_excel(data_list[0]).columns
df = pd.DataFrame(columns=col)
for i in range(len(data_list)):
    one_table = pd.read_excel(data_list[i])
    df = df.append(one_table)

#df.to_csv('../data/eda_data.csv', encoding = 'utf-8-sig', index= False)    

#%% 컬럼 선택, 중복 제거 그리고 정렬까지 연습
df = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df = df.drop_duplicates()
df = df.sort_values(by=['검사명', '검체'], ascending=[True, True], axis= 0)
#2880 rows

#%% 잘못된 데이터 한번에 수정하기
#df.to_csv('../data/eda_data.csv', encoding = 'utf-8-sig')
for i in range(13,55):
    one_table = pd.read_excel(data_list[i])
    one_table.loc[one_table['검사명'] == 'Micro Alb Ratio', '단위'] = 'ug/mg'
    one_table.to_excel(data_list[i], encoding = 'utf-8-sig', index= False)    

#%% EDA
df_0 = df[['검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df_0 = df_0.sort_values(by=['검사명'], ascending=[True], axis= 0)
#rows 218

df_1 = df[['검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df_1 = df_1.sort_values(by=['검사명'], ascending=[True], axis= 0)
#rows 249

df_2 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df_2 = df_2.sort_values(by=['검사명'], ascending=[True], axis= 0)
df_2_count = df_2.groupby(by = ['검체', '검체검사결과', '검사명']).count()
df_2_count = df_2_count.reset_index()
df_2_count.to_csv('../data/groupby_검체_검체검사결과_검사명.csv', index=False, encoding='utf-8-sig')
#rows 251 
#PT- sec는 2개씩, Free T4, T3, TSH는 3개씩 중복됨.
df_2 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df_2_countt = df_2.groupby(by = ['검체검사결과', '검사명']).count()
# 중복
df_2_counttt = df_2.groupby(by = ['검체',  '검사명']).count()
#너무 중복되는게 많음
#결론
# 검체, 검체검사결과, 검사명을 모두 포함한 복합키일 경우에 참고치와 단위를 정확하게 구분 할 수 있다.

df_3 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위', '검체번호']].drop_duplicates()
df_3 = df_3.sort_values(by=['검사명'], ascending=[True], axis= 0)
#rows 2858 #검체번호 x 검사명

df_4 = df[['검체']].drop_duplicates()
#rows 14

df_5 = df[['검체검사결과']].drop_duplicates()
#rows 23    

df_6 = df[['검체번호']].drop_duplicates()
#rows 232

df_7 = df[['검체번호', '검사명']].drop_duplicates()
df_7 = df_7.sort_values(by=['검사명'], ascending=[True], axis= 0)
#rows 2858
#중복되는 검체번호는 3개 있음 # 종이기록으로 확인함.

df_8 = df[['검체번호', '검체검사결과']].drop_duplicates()
df_8_count = df_8.groupby(by = '검체번호').count()
#rows 234

df_9 = df[['검체번호', '검체']].drop_duplicates()
df_9_count = df_9.groupby(by = '검체번호').count()
df_9_countt = df_9.groupby(by = '검체').count()
#rows 233

dff_9 = df[['검체검사결과', '검체']].drop_duplicates().sort_values(by=['검체검사결과'], ascending=[True], axis= 0)


dff_10 = df[['검체번호', '검체검사결과', '검체']].drop_duplicates()
#rows 234

dff_11 = df[['검체번호', '검체검사결과', '검사명']].drop_duplicates()
#rows 2858 #검체번호 x 검사명


#%%
df_doc = df[['검체번호', '의뢰의사','검사자']].drop_duplicates()
df_doc_groupby = df_doc.groupby(by = ['검체번호']).count()
df_doc_groupby = df_doc_groupby.reset_index()

df_doc.iloc[139] #데이터를 보면 다 중복 없는데..?
df_doc.iloc[49]
df_doc.iloc[86]

df_doc.iloc[138] #얘는 중복있네
df_doc.iloc[48]
df_doc.iloc[85]


# 검체 >>> 검체검사결과
# 검체 어떤 물질을 어떤 방식으로 검사했는지 ex) Blood(EDTA)
# 검체검사결과 검체한것중 어느 결과를 볼것인지 ex)간염,종양,빈혈및기타, 단백면역화학, 약물및중금속, 응급검사, 일반혈액, 특수화학검사, 혈액은행검사
#%%
df_검체_검사명 = df[['검체', '검사명']].drop_duplicates().sort_values(by=['검체', '검사명'], ascending=[True, True], axis= 0)
#213
df_검체검사결과_검사명 = df[['검체검사결과', '검사명']].drop_duplicates().sort_values(by=['검체검사결과', '검사명'], ascending=[True, True], axis= 0)
#240
df_검체_검사명_검체검사결과 = df[['검체', '검체검사결과', '검사명']].drop_duplicates().sort_values(by=['검체', '검체검사결과', '검사명'], ascending=[True, True, True], axis= 0)
#242 2rows added 
#1. Cuture 일반미생물 Sputum Urine (Void)
#2. Gram Stain 일반미생물 Sputum Urine (Void)

df_5 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates().sort_values(by=['검체', '검체검사결과', '검사명'], ascending=[True, True, True], axis= 0)
#250 8row added 
#갑상선종양검사 6rows added
#PT- sec 1row added
#Micro Alb Ratio 1row added
#%%
df_full_검체번호 = df[['검체번호', '접수일', '의뢰일', '결과보고일', '의뢰의사', '검사자']].drop_duplicates()
#238
df_검체번호 = df[['검체번호']].drop_duplicates()
#232
df_No_검체번호 = df[['접수일', '의뢰일', '결과보고일', '의뢰의사', '검사자']].drop_duplicates()
#209

#%% 
df_5 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates().sort_values(by=['검체', '검체검사결과', '검사명'], ascending=[True, True, True], axis= 0)
df_5.to_csv('../data/compare_data.csv', encoding = 'utf-8-sig', index= False)    


#%% 잘못된 데이터 변경하기 a -> b 
data_list = glob('D:/05.Kidney/data/개별데이터/*.xlsx')
for num, name in enumerate(data_list):
    print(num, name)
    one_table = pd.read_excel(data_list[num])
    one_table.loc[one_table['검사명'] == 'HDL-C', '최저참고치'] = 35
    one_table.to_excel('{}'.format(name), encoding = 'utf-8-sig', index= False)    
    
    
    
    
    
    
    