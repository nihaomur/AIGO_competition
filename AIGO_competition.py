# %% [markdown]
# # Import Module

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %% [markdown]
# # Read data

# %%
train = pd.read_csv('training_data.csv')
test = pd.read_csv('public_dataset.csv')
submit = pd.read_csv('public_submission_template.csv')

# %%
train.shape

# %%
train.head()

# %%
train.info()

# %%
test.head()

# %%
test.info()

# %% [markdown]
# # Concatate the data and do feature engineering.

# %%
data = pd.concat([train, test])
data.reset_index(inplace=True, drop=True)
data

# %%
print(data['單價'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(data['單價'], color='g', bins=100, hist_kws={'alpha': 0.4})

# %%
import scipy.stats as stats

y = data['單價']
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=stats.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=stats.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=stats.lognorm)

# %% [markdown]
#     看起來在去識別化時已經幫我們調整過，對預測標的「單價」不需多做調整

# %% [markdown]
# ### 備註跟分區太沒用太少了，drop掉。

# %%
data.drop('使用分區', axis=1, inplace=True)
data.drop('備註', axis=1, inplace=True)
data.drop('ID', axis=1, inplace=True)

# %%
data.columns

# %% [markdown]
# ### 「地址」，從內政部抓取實價登錄資料，以近一年的「元/每平方公尺」來代替。

# %% [markdown]
# - 112年第3季的實價登錄資料。

# %%
import os
import pandas as pd

folder_path = '112_3'  #read 112_3 folder

combined_data_112_3 = pd.DataFrame() #create a empty df to concat all file

for filename in os.listdir(folder_path):
    if filename.endswith("lvr_land_a.csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        combined_data_112_3 = pd.concat([combined_data_112_3, df], ignore_index=True)

# %%
combined_data_112_3.shape #(90866,33) same as 實價登錄's columns numbers

# %%
import os
import pandas as pd

folder_path = '112_2'  #read 112_2 folder

combined_data_112_2 = pd.DataFrame() 

for filename in os.listdir(folder_path):
    if filename.endswith("lvr_land_a.csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        combined_data_112_2 = pd.concat([combined_data_112_2, df], ignore_index=True)
        

# %%
combined_data_112_2.shape #(89280,33)

# %%
import os
import pandas as pd

folder_path = '112_1'  #read 112_2 folder

combined_data_112_1 = pd.DataFrame() 

for filename in os.listdir(folder_path):
    if filename.endswith("lvr_land_a.csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        combined_data_112_1 = pd.concat([combined_data_112_1, df], ignore_index=True)
        

# %%
combined_data_112_1.shape #(76316, 33)

# %%
import os
import pandas as pd

folder_path = '111_4'  #read 112_2 folder

combined_data_111_4 = pd.DataFrame() 

for filename in os.listdir(folder_path):
    if filename.endswith("lvr_land_a.csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        combined_data_111_4 = pd.concat([combined_data_111_4, df], ignore_index=True)
        

# %%
combined_data_111_4.shape #(87341, 33)

# %% [markdown]
# - 將四個檔案連結起來。

# %%
df_trans = pd.concat([combined_data_112_3, combined_data_112_2, combined_data_112_1, combined_data_111_4], ignore_index=True)
df_trans.shape

# %% [markdown]
# - 抓取 df_trans 中的兩欄位：「土地位置建物門牌」、「單價元平方公尺」

# %%
df_trans = df_trans[['土地位置建物門牌','單價元平方公尺']]
df_trans

# %% [markdown]
# - 訓練及預測檔案皆只有包含「房屋」，不包含純土地買賣，將參考訓練df_trans中的「地號」（即為土地），全部刪去，

# %%
df_trans = df_trans[~df_trans['土地位置建物門牌'].str.endswith('地號')]
df_trans

# %% [markdown]
# - 將空值欄位刪去

# %%
df_trans.dropna(inplace=True)
df_trans

# %% [markdown]
# - 合併資料集中的「'縣市', '鄉鎮市區', '路名'」為一 col

# %%
data['建物地址'] = data['縣市'] + data['鄉鎮市區'] + data['路名']
data.drop(['縣市', '鄉鎮市區', '路名'], axis=1, inplace=True)
data

# %%
df_trans = df_trans.drop(df_trans.index[0])
df_trans

# %%
df_trans.tail()

# %%
average_prices = []

# 遍歷data，輸出相應df_trans中包含地址的平均地價。
for address in data["建物地址"]:
    
    filtered_data_trans = df_trans[df_trans["土地位置建物門牌"].str.contains(address, case=False, na=False)]

    filtered_data_trans["單價元平方公尺"] = pd.to_numeric(filtered_data_trans["單價元平方公尺"], errors='coerce')

    average_price = filtered_data_trans["單價元平方公尺"].mean()

    average_prices.append(average_price)

average_prices


# %% [markdown]
# - 將average_prices輸出成csv檔，以利保存。

# %%
average_prices_series = pd.Series(average_prices, name="平均單價元平方公尺")

csv_file = "average_prices.csv"

average_prices_series.to_csv(csv_file, index=False)

# %%
average_prices_series.shape

# %%
average_prices_series.isna().sum() #大約有1/3的值是空值，需要填入相應空值。

# %%
print(average_prices_series.describe())
plt.figure(figsize=(9, 8))
sns.distplot(average_prices_series, color='g', bins=100, hist_kws={'alpha': 0.4})


# %%
data = pd.concat([data, average_prices_series], axis = 1)
data.head()

# %% [markdown]
# - 將橫坐標、縱座標利用函式轉換為經緯度。
# 
# - source : https://tylerastro.medium.com/twd97-to-longitude-latitude-dde820d83405

# %%
import math

def twd97_to_lonlat(x=174458.0,y=2525824.0):
    """
    Parameters
    ----------
    x : float
        TWD97 coord system. The default is 174458.0.
    y : float
        TWD97 coord system. The default is 2525824.0.
    Returns
    -------
    list
        [longitude, latitude]
    """
    
    a = 6378137
    b = 6356752.314245
    long_0 = 121 * math.pi / 180.0
    k0 = 0.9999
    dx = 250000
    dy = 0
    
    e = math.pow((1-math.pow(b, 2)/math.pow(a,2)), 0.5)
    
    x -= dx
    y -= dy
    
    M = y / k0
    
    mu = M / ( a*(1-math.pow(e, 2)/4 - 3*math.pow(e,4)/64 - 5 * math.pow(e, 6)/256))
    e1 = (1.0 - pow((1   - pow(e, 2)), 0.5)) / (1.0 +math.pow((1.0 -math.pow(e,2)), 0.5))
    
    j1 = 3*e1/2-27*math.pow(e1,3)/32
    j2 = 21 * math.pow(e1,2)/16 - 55 * math.pow(e1, 4)/32
    j3 = 151 * math.pow(e1, 3)/96
    j4 = 1097 * math.pow(e1, 4)/512
    
    fp = mu + j1 * math.sin(2*mu) + j2 * math.sin(4* mu) + j3 * math.sin(6*mu) + j4 * math.sin(8* mu)
    
    e2 = math.pow((e*a/b),2)
    c1 = math.pow(e2*math.cos(fp),2)
    t1 = math.pow(math.tan(fp),2)
    r1 = a * (1-math.pow(e,2)) / math.pow( (1-math.pow(e,2)* math.pow(math.sin(fp),2)), (3/2))
    n1 = a / math.pow((1-math.pow(e,2)*math.pow(math.sin(fp),2)),0.5)
    d = x / (n1*k0)
    
    q1 = n1* math.tan(fp) / r1
    q2 = math.pow(d,2)/2
    q3 = ( 5 + 3 * t1 + 10 * c1 - 4 * math.pow(c1,2) - 9 * e2 ) * math.pow(d,4)/24
    q4 = (61 + 90 * t1 + 298 * c1 + 45 * math.pow(t1,2) - 3 * math.pow(c1,2) - 252 * e2) * math.pow(d,6)/720
    lat = fp - q1 * (q2 - q3 + q4)
    
    
    q5 = d
    q6 = (1+2*t1+c1) * math.pow(d,3) / 6
    q7 = (5 - 2 * c1 + 28 * t1 - 3 * math.pow(c1,2) + 8 * e2 + 24 * math.pow(t1,2)) * math.pow(d,5) / 120
    lon = long_0 + (q5 - q6 + q7) / math.cos(fp)
    
    lat = (lat*180) / math.pi
    lon = (lon*180) / math.pi
    return [lon, lat]

twd97_to_lonlat(x = 305266, y = 2768378)

# %%
df_ll = pd.DataFrame(zip(data['橫坐標'],data['縱坐標']))
print(df_ll)

# %%
result = df_ll.apply(lambda row: pd.Series(twd97_to_lonlat(row[0], row[1])), axis=1)
result = result.rename(columns={0: 'lng', 1: 'lat'})
result

# %%
data = pd.concat([data,result],axis=1)
data.head()

# %%
data.drop(['橫坐標','縱坐標'], axis=1, inplace= True)
data.head()

# %%
data.shape

# %% [markdown]
# ### 對各個類別特徵做 one-hot encoding

# %%
data['主要用途'].value_counts()

# %% [markdown]
# 一般事務所、辦公室、店舖併進商業用，國民住宅、住商、住工併進住家用，工業用、廠房併進其他。

# %%
apply_map = {'住家用':'住家用', '國民住宅':'住家用', '住商用':'住家用', '住工用':'住家用', 
             '集合住宅':'集合住宅', 
             '其他':'其他', '工業用':'其他', '廠房':'其他',
             '商業用':'商業用', '一般事務所':'商業用', '辦公室':'商業用', '店鋪':'商業用'}

data['主要用途'] = data['主要用途'].map(apply_map)
data['主要用途'].value_counts()

# %%
data = pd.get_dummies(data, columns=['主要用途'])

# %%
data.columns

# %% [markdown]
# - 主要建材的 one-hot encoding

# %%
data['主要建材'].value_counts()

# %% [markdown]
# 鋼筋混凝土加強磚造併進鋼筋混凝土造，磚造併進加強磚造。

# %%
material_map = {'鋼筋混凝土造':'鋼筋混凝土造', '鋼筋混凝土加強磚造':'鋼筋混凝土造', 
                '鋼骨造':'鋼骨造', 
                '加強磚造':'加強磚造', '磚造':'加強磚造',
                '其他':'其他'}

data['主要建材'] = data['主要建材'].map(material_map)
data['主要建材'].value_counts()

# %%
data = pd.get_dummies(data, columns=['主要建材'])

# %%
data.columns

# %% [markdown]
# - 發現建物型態與樓層數高度相關，drop掉避免互相影響。

# %%
data['建物型態'].value_counts()

# %%
data.drop('建物型態' , axis=1, inplace=True)

# %%
data.info()

# %% [markdown]
# ## 經緯度連接外部資料集：
#     選擇以下特徵： 醫療機構（只取醫院以上）； 捷運站 ； 火車站 ； 國小、國中、高中、大學。

# %% [markdown]
# - 醫院：方圓三公里內有無。
# 
# - 捷運站：一公里內有幾個捷運站。
# 
# - 火車站：三公里內有無。
# 
# - 學區：三公里有幾個學校。

# %%
pip install geopy

# %% [markdown]
#     醫院

# %%
data_hospital = pd.read_csv('external_data/醫療機構基本資料.csv')
data_hospital.columns

# %%
data_hospital['型態別'].value_counts()

# %% [markdown]
#     將其中的醫院、綜合醫院等篩選出來，保留他們的位置「lat」、「lng」。

# %%
filter_hospital = ['醫院','綜合醫院']
data_hospital = data_hospital[data_hospital['型態別'].isin(filter_hospital)]

# %%
data_hospital['型態別'].value_counts()

# %%
#只保留lat, lng
data_hospital = data_hospital[['lat','lng']]
data_hospital

# %%
data[['lat','lng']]

# %% [markdown]
# ### 載入經緯度距離套件，並且計算3公里是否有醫療機構。

# %%
from geopy.distance import geodesic

# 假设你有两个数据集 data 和 contrast，分别包含经度和纬度列
# 创建一个空的新列用于存储结果
data['hospital_within_1km'] = False

# 遍历 data 数据集
for index, row_data in data.iterrows():
    lat1, lng1 = row_data['lat'], row_data['lng']
    
    # 遍历 contrast 数据集
    for _, row_contrast in data_hospital.iterrows():
        lat2, lng2 = row_contrast['lat'], row_contrast['lng']
        
        # 计算两点之间的距离
        distance = geodesic((lat1, lng1), (lat2, lng2)).kilometers
        
        # 如果距离小于等于1公里，则将 within_1km 列设置为 True
        if distance <= 1:
            data.at[index, 'hospital_within_1km'] = True
            break  # 如果已经找到一个在一公里内的点，可以跳出内层循环

data['hospital_within_1km'].value_counts()


# %%
print(data[data['hospital_within_1km']==True]['單價'].describe())
print('-'*50)
print(data[data['hospital_within_1km']==False]['單價'].describe())

# %%
print('一公里內有醫院')
print(data[data['hospital_within_1km']==True]['平均單價元平方公尺'].describe())
print('-'*50)
print('一公里內無醫院')
print(data[data['hospital_within_1km']==False]['平均單價元平方公尺'].describe())

# %%
data.columns

# %%
data['hospital_within_1km'].value_counts()

# %% [markdown]
# 將醫院檔案輸出以利保存

# %%
hospital_series = pd.Series(data['hospital_within_1km'], name='hospital_within_1km')

csv_file = "hospital_in_1km.csv"

hospital_series.to_csv(csv_file, index=False)

# %% [markdown]
#     方圓三公里內有無火車站

# %%
data_railway = pd.read_csv('external_data/火車站點資料.csv')
data_railway.columns

# %% [markdown]
# 只取三等站以上的站別

# %%
data_railway = data_railway[data_railway['車站級別'] < 4]
data_railway['車站級別'].value_counts()

# %%
data_railway = data_railway[['lat','lng']]
data_railway

# %%
data[['lat','lng']]

# %%
data.columns

# %%
data['railway_within_0.5km'] = False

# 遍历 data 数据集
for index, row_data in data.iterrows():
    lat1, lng1 = row_data['lat'], row_data['lng']
    
    # 遍历 contrast 数据集
    for _, row_contrast in data_railway.iterrows():
        lat2, lng2 = row_contrast['lat'], row_contrast['lng']
        
        # 计算两点之间的距离
        distance = geodesic((lat1, lng1), (lat2, lng2)).kilometers
        
        # 如果距离小于等于1公里，则将 within_1km 列设置为 True
        if distance <= 0.5:
            data.at[index, 'railway_within_0.5km'] = True
            break  # 如果已经找到一个在一公里内的点，可以跳出内层循环

data['railway_within_0.5km'].value_counts()


# %%
print(data['railway_within_0.5km'].value_counts())
print(data['railway_within_1km'].value_counts())
print(data['railway_within_1.5km'].value_counts())
print(data['railway_within_2km'].value_counts())
print(data['railway_within_3km'].value_counts())

# %%
print('2公里內有車站的單價')
print(data[data['railway_within_2km']==True]['單價'].describe())
print('-'*50)
print('2公里內無車站的單價')
print(data[data['railway_within_2km']==False]['單價'].describe())

# %%
average_prices = data.groupby('railway_within_2km')['單價'].mean()

plt.bar(average_prices.index, average_prices)

plt.xlabel('railway_within_2km')
plt.ylabel('平均單價')
plt.title('不同 railway_within_2km 值的平均單價')


plt.show()

# %%
print('3公里內有車站的單價')
print(data[data['railway_within_3km']==True]['單價'].describe())
print('-'*50)
print('3公里內無車站的單價')
print(data[data['railway_within_3km']==False]['單價'].describe())

# %%
print('1.5公里內有車站的單價')
print(data[data['railway_within_1.5km']==True]['單價'].describe())
print('-'*50)
print('1.5公里內無車站的單價')
print(data[data['railway_within_1.5km']==False]['單價'].describe())

# %%
print('1公里內有車站的單價')
print(data[data['railway_within_1km']==True]['單價'].describe())
print('-'*50)
print('1公里內無車站的單價')
print(data[data['railway_within_1km']==False]['單價'].describe())

# %%
print('0.5公里內有車站的單價')
print(data[data['railway_within_0.5km']==True]['單價'].describe())
print('-'*50)
print('0.5公里內無車站的單價')
print(data[data['railway_within_0.5km']==False]['單價'].describe())

# %% [markdown]
# 將車站檔案輸出以利保存

# %%
railway_series = pd.Series(data['railway_within_0.5km'], name='railway_within_0.5km')
csv_file = "railway_in_0.5km.csv"
railway_series.to_csv(csv_file, index=False)

railway_series = pd.Series(data['railway_within_1km'], name='railway_within_1km')
csv_file = "railway_in_1km.csv"
railway_series.to_csv(csv_file, index=False)

railway_series = pd.Series(data['railway_within_1.5km'], name='railway_within_1.5km')
csv_file = "railway_in_1.5km.csv"
railway_series.to_csv(csv_file, index=False)

railway_series = pd.Series(data['railway_within_2km'], name='railway_within_2km')
csv_file = "railway_in_2km.csv"
railway_series.to_csv(csv_file, index=False)

railway_series = pd.Series(data['railway_within_3km'], name='railway_within_3km')
csv_file = "railway_in_3km.csv"
railway_series.to_csv(csv_file, index=False)

# %%
data.drop(['railway_within_0.5km', 'railway_within_1km','railway_within_1.5km', 'railway_within_3km'], axis=1, inplace=True)

# %%
data.columns

# %%
data.head()

# %% [markdown]
# 捷運站資料中只有：台北市、新北市、桃園市、高雄市（無台中市）

# %%
data_mrt = pd.read_csv('external_data/捷運站點資料.csv')
data_mrt.columns

# %%
data_mrt['站點地址'].str[0:3].unique() #沒有台中

# %%
data_mrt = data_mrt[['lat','lng']]
data_mrt

# %%
data['建物地址'].str[0:3].unique()

# %%
data['mrt_numbs_within_0.5km'] = 0
mrt_cities = ['台北市', '新北市', '桃園市', '高雄市']

# 遍历 data 数据集
for index, row_data in data.iterrows():
    if row_data['建物地址'][:3] in mrt_cities:
        lat1, lng1 = row_data['lat'], row_data['lng']
        mrt_count = 0
    # 遍历 contrast 数据集
        for _, row_contrast in data_mrt.iterrows():
            lat2, lng2 = row_contrast['lat'], row_contrast['lng']
        
        # 计算两点之间的距离
            distance = geodesic((lat1, lng1), (lat2, lng2)).kilometers
        
        # 如果距离小于等于1公里，则将 within_1km 列设置为 True
            if distance <= 0.5:
                mrt_count += 1
        
        data.at[index, 'mrt_numbs_within_0.5km'] = mrt_count
    
data['mrt_numbs_within_1km'].value_counts()


# %%
data['mrt_numbs_within_0.5km'].value_counts()

# %% [markdown]
# 將捷運資料保存

# %%
railway_series = pd.Series(data['mrt_numbs_within_1km'], name='mrt_numbs_within_1km')
csv_file = "mrt_numbs_within_1km.csv"
railway_series.to_csv(csv_file, index=False)

railway_series = pd.Series(data['mrt_numbs_within_0.5km'], name='mrt_numbs_within_0.5km')
csv_file = "mrt_numbs_within_0.5km.csv"
railway_series.to_csv(csv_file, index=False)

# %%
pd.read_csv('mrt_numbs_within_0.5km.csv').value_counts()

# %%
data[data['mrt_numbs_within_1km']>8]['單價'].describe()

# %%
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

average_prices = data.groupby('mrt_numbs_within_0.5km')['單價'].mean()

plt.bar(average_prices.index, average_prices)

plt.xlabel('mrt_numbs_within_0.5km')
plt.ylabel('平均單價')
plt.title('不同 mrt_numbs_within_0.5km 值的平均單價')


plt.show()


# %%
average_prices = data.groupby('mrt_numbs_within_1km')['單價'].mean()

plt.bar(average_prices.index, average_prices)

plt.xlabel('mrt_numbs_within_1km')
plt.ylabel('平均單價')
plt.title('不同 mrt_numbs_within_1km 值的平均單價')


plt.show()

# %%
data.columns

# %%
elemenatry = pd.read_csv('external_data/國小基本資料.csv')
senior = pd.read_csv('external_data/國中基本資料.csv')
high = pd.read_csv('external_data/高中基本資料.csv')
college = pd.read_csv('external_data/大學基本資料.csv')

elemenatry = elemenatry[['lat','lng']]
senior = senior[['lat','lng']]
high = high[['lat','lng']]
college = college[['lat','lng']]

data_schools = pd.concat([elemenatry, senior, high, college], axis=0, ignore_index = True)

data_schools


# %%
data['school_numbs_within_1km'] = 0

# 遍历 data 数据集
for index, row_data in data.iterrows():
        lat1, lng1 = row_data['lat'], row_data['lng']
        school_count = 0
    # 遍历 contrast 数据集
        for _, row_contrast in data_schools.iterrows():
            lat2, lng2 = row_contrast['lat'], row_contrast['lng']
        
        # 计算两点之间的距离
            distance = geodesic((lat1, lng1), (lat2, lng2)).kilometers
        
        # 如果距离小于等于1公里，则将 within_1km 列设置为 True
            if distance <= 1:
                school_count += 1
        
        data.at[index, 'school_numbs_within_1km'] = school_count
    
data['school_numbs_within_1km'].value_counts()


# %% [markdown]
# 

# %%
average_prices = data.groupby('school_numbs_within_1km')['單價'].mean()

plt.bar(average_prices.index, average_prices)

plt.xlabel('school_numbs_within_1km')
plt.ylabel('平均單價')
plt.title('不同 school_numbs_within_1km 值的平均單價')


plt.show()

# %% [markdown]
# 將學校資料保存

# %%
school_series = pd.Series(data['school_numbs_within_1km'], name='school_numbs_within_1km')
csv_file = "school_numbs_within_1km.csv"
school_series.to_csv(csv_file, index=False)

# %% [markdown]
# 檢查整個data的資訊

# %%
data.info()

# %% [markdown]
# 「平均單價元平方公尺」存在空值，對其進行KNN補值，
# 
# 在此之前需要先分離「單價」，以免應變數反過來影響自變數;
# 
# 確認檔案內都是數值，Bool要改成0/1。

# %%
data.dtypes

# %%
bool_columns = data.select_dtypes(include=['bool']).columns

data[bool_columns] = data[bool_columns].astype(int)

data.dtypes

# %%
data.iloc[:,17:23].head()

# %%
price = data['單價']
data.drop('單價',axis=1,inplace=True)


# %%
from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5)
data['平均單價元平方公尺'] = knn_imputer.fit_transform(data[['平均單價元平方公尺']])

# %%
data['平均單價元平方公尺'].describe()

# %% [markdown]
# 至此，經緯度已經可以拋棄了。

# %%
data.drop(['lat','lng'], axis=1, inplace=True)

# %% [markdown]
# 將單價併回在最後一欄位，以方便我們進行模型訓練。

# %%
data = pd.concat([data,price],axis=1)

# %% [markdown]
# 檢查data

# %%
print(data.columns)
print('-'*50)
print(data.info())

# %% [markdown]
# 還存在一個非可轉換數值欄位：建物地址，drop掉

# %%
print('-'*50)
print(data.dtypes.value_counts())

# %%
data.drop('建物地址',axis=1,inplace=True)

# %%
data.head()

# %% [markdown]
# 保存data(畢竟花超久才整理好)

# %%
data_1102 = data
csv_file = "data_1102.csv"
data_1102.to_csv(csv_file, index=False)

# %% [markdown]
# 最後，將train_data與submit_data拆開。

# %%
data = pd.read_csv('data_1102.csv')

# %%
data.columns

# %%
data.info()

# %%
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()

# %% [markdown]
# 將與「單價」相關係數太低（低於0.03），或是特徵間有高度線性相關的特徵捨去（主建物面積與建物面積、陽台面積；車位個數與車位面積）

# %%
low_cor = ['建物面積','陽台面積','車位個數','主要用途_住家用','主要用途_其他','railway_within_2km']
data.drop(low_cor, axis=1, inplace=True)
data.columns

# %%


# %%
corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()

# %%
dataTrain = data.iloc[:11751, :]
dataSubmit = data.iloc[11751:, :-1]
dataTrain.tail()

# %%
dataSubmit.head()

# %% [markdown]
# 終於，我們將特徵工程處理完畢，接著選定模型進行訓練。

# %%
data = pd.read_csv('dataset_1109_public.csv')
data.info()

# %%
data.head()

# %%
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()

# %%
data.drop('atm_numbs_within_0.5km', axis=1 ,inplace=True)
data.drop('平均單價元平方公尺', axis=1 ,inplace=True)

# %%
data.info()

# %%
price_pu = pd.read_csv('average_prices_5years_pu_1113_mean.csv')
data.insert(22,'平均單價元平方公尺', price_pu)

# %%
price_pu

# %%
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()

# %%
data.drop('縣市區域價格',axis=1,inplace=True)

# %%
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=70)
plt.yticks(rotation=0)

plt.show()

# %%
#oldone

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
corr_matrix = data.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()

# %%
data

# %%
dataTrain = data.iloc[:11751, :]
dataSubmit = data.iloc[11751:, :-1]
dataTrain.tail()

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold


# %%
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


# %%
X, y = dataTrain.iloc[:,:-1], dataTrain.iloc[:,-1]

# %%
X.shape

# %% [markdown]
#     用Random Forest Model 進行訓練，並以網格搜尋找出最好的超參數。

# %%
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


pipe_rf = make_pipeline(RobustScaler(), RandomForestRegressor(random_state=1))


mae_scores_rf = -cross_val_score(pipe_rf, X, y, scoring='neg_mean_absolute_error', cv=kfolds, n_jobs=2)

print('Random Forest MAE:', mae_scores_rf)
print('Random Forest 平均 MAE:', mae_scores_rf.mean())

# %%
#1113
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np


param_grid_rf = {
    'randomforestregressor__n_estimators': [1000, 1200],
    'randomforestregressor__max_depth': [10, None],
}


gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=param_grid_rf,
                     scoring='neg_mean_absolute_error',
                     cv=kfolds,
                     n_jobs=-1)


gs_rf.fit(X, y)


print('Best Parameters:', gs_rf.best_params_)
print('Best MAE:', -gs_rf.best_score_)


# %%
#1113
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np


param_grid_rf = {
    'randomforestregressor__n_estimators': [1000, 1200],
    'randomforestregressor__max_depth': [10, None],
}


gs_rf = GridSearchCV(estimator=pipe_rf,
                     param_grid=param_grid_rf,
                     scoring='neg_mean_absolute_error',
                     cv=kfolds,
                     n_jobs=-1)


gs_rf.fit(X, y)


print('Best Parameters:', gs_rf.best_params_)
print('Best MAE:', -gs_rf.best_score_)

# %% [markdown]
#     XGBoost 模型訓練

# %%
##1113

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


pipe_xgb = make_pipeline(RobustScaler(), XGBRegressor(random_state=1))

mae_scores_xgb = -cross_val_score(pipe_xgb, X, y, scoring='neg_mean_absolute_error', cv=kfolds)

print('XGB MAE:', mae_scores_xgb)
print('XGB 平均 MAE:', mae_scores_xgb.mean())

# %%
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import numpy as np

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

pipe_xgb = make_pipeline(RobustScaler(), XGBRegressor(random_state=1))

param_grid_xgb = {
    'xgbregressor__n_estimators': [1000, 1200],
    'xgbregressor__max_depth': [8, 10],
    'xgbregressor__learning_rate': [0.07]
}


gs_xgb = GridSearchCV(estimator=pipe_xgb,
                     param_grid=param_grid_xgb,
                     scoring='neg_mean_absolute_error',
                     cv=kfolds,
                     n_jobs=-1)

gs_xgb.fit(X, y)


print('Best Parameters:', gs_xgb.best_params_)
print('Best MAE:', -gs_xgb.best_score_)


# %% [markdown]
#     Catboost 模型預測

# %%
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold

# 初始化CatBoostRegressor模型
catboost_model = CatBoostRegressor(depth=7, iterations=15000, learning_rate=0.07, loss_function='MAE')



# 执行交叉验证并获取性能指标
scores = cross_val_score(catboost_model, X, y, scoring='neg_mean_absolute_error')

# 打印交叉验证得分
print("交叉验证MAE得分：", -scores)



# %%
cvcat = np.array([0.17318409, 0.17752437, 0.17617373, 0.18059843, 0.17156367])
cvcat.mean()

# %%
from catboost import CatBoostRegressor, Pool, cv

# 创建CatBoost回归模型
catboost_model = CatBoostRegressor(learning_rate=0.05, loss_function='MAE')

# 创建数据集
catboost_data = Pool(X, label=y)

# 设置要调整的超参数范围
param_grid_catboost = {
    'iterations': [15000, 18000],
    'depth': [7],
    'learning_rate': [0.06, 0.07]
}

# 执行交叉验证和超参数搜索
grid_catboost = catboost_model.grid_search(param_grid_catboost, catboost_data, partition_random_seed=1, verbose=0)


# %%
#1113
from catboost import CatBoostRegressor, Pool, cv

# 创建CatBoost回归模型
catboost_model = CatBoostRegressor(learning_rate=0.05, loss_function='MAE')

# 创建数据集
catboost_data = Pool(X, label=y)

# 设置要调整的超参数范围
param_grid_catboost = {
    'iterations': [8000, 12000],
    'depth': [6, 7],
    'learning_rate': [0.06, 0.07]
}

# 执行交叉验证和超参数搜索
grid_catboost = catboost_model.grid_search(param_grid_catboost, catboost_data, partition_random_seed=1, verbose=0)


# %%
# 1113 last
best_params = grid_catboost['params']
best_mae = min(grid_catboost['cv_results']['test-MAE-mean'])

print('Best Parameters:', best_params)
print('Best MAE:', best_mae)

# %%
# 1113
best_params = grid_catboost['params']
best_mae = min(grid_catboost['cv_results']['test-MAE-mean'])

print('Best Parameters:', best_params)
print('Best MAE:', best_mae)

# %%
# 获取最佳参数和MAE得分
best_params = grid_catboost['params']
best_mae = min(grid_catboost['cv_results']['test-MAE-mean'])

print('Best Parameters:', best_params)
print('Best MAE:', best_mae)

# %%
best_catboost_model = CatBoostRegressor(**best_params)
best_catboost_model.fit(X,y)

# %%
dataSubmit.info()

# %%
dataSubmit.columns

# %%
dataPrivate = pd.read_csv('dataset_private_1109.csv')
dataPrivate.columns

# %%
dataPrivate.drop('atm_numbs_within_0.5km', axis=1, inplace=True)
dataPrivate.drop('平均單價元平方公尺', axis=1, inplace=True)

# %%
price_pr = pd.read_csv('average_prices_5years_pr_1113_mean.csv')
dataPrivate.insert(22,'平均單價元平方公尺', price_pr)

# %%
dataPrivate.drop('縣市區域價格', axis=1, inplace=True)

# %%
dataPrivate.columns

# %%
# create data for catboost
catboost_data_submit = Pool(dataSubmit)

predictions_catboost_submit = best_catboost_model.predict(catboost_data_submit)

print(predictions_catboost_submit)

# %%
#1113
catboost_data_submit = Pool(dataSubmit)

predictions_catboost_submit = best_catboost_model.predict(catboost_data_submit)

print(predictions_catboost_submit)

# %%
# create data for catboost
catboost_data_private = Pool(dataPrivate)

predictions_catboost_private = best_catboost_model.predict(catboost_data_private)

print(predictions_catboost_private)

# %%
# 1113
catboost_data_private = Pool(dataPrivate)

predictions_catboost_private = best_catboost_model.predict(catboost_data_private)

print(predictions_catboost_private)

# %%
pd.Series(predictions_catboost_submit).describe()

# %%
pd.Series(predictions_catboost_submit).describe()

# %%
pd.Series(predictions_catboost_private).describe()

# %%
submit_cat = pd.concat([pd.Series(predictions_catboost_submit), pd.Series(predictions_catboost_private)], axis=0)

# %%
submit_cat = np.array(submit_cat).reshape(-1)

# %%
submit_cat

# %%
X.shape

# %%
submit_rfgs = gs_rf.predict(dataSubmit)
submit_xgbgs = gs_xgb.predict(dataSubmit)

# %%
submit_rfgs #1111last

# %%
submit_rfgs #1113

# %%
submit_xgbgs #1111 last

# %%
submit_xgbgs #1113

# %%
private_rfgs = gs_rf.predict(dataPrivate)
private_xgbgs = gs_xgb.predict(dataPrivate)

# %%
private_xgbgs #1111last

# %%
private_xgbgs #1113

# %%
submit_lr = 0.3 * submit_xgbgs + 0.0 * submit_rfgs + 0.7 * predictions_catboost_submit #1113

# %%
submit_lr = 0.4 * submit_xgbgs + 0.1 * submit_rfgs + 0.5 * predictions_catboost_submit 

# %%
private_lr = 0.3 * private_xgbgs + 0.0 * private_rfgs + 0.7 * predictions_catboost_private #1113

# %%
pd.Series(private_xgbgs).describe()

# %%
pd.Series(submit_xgbgs).describe()

# %%
submit_lr #1111last

# %%
#1113
submit_lr

# %%
private_lr #1111 last

# %%
private_lr #1113

# %%
data['單價'].describe()

# %%
#1113
print(pd.Series(submit_lr).describe())
print(pd.Series(submit_lr).shape[0])
print('-'*50)
print(pd.Series(private_lr).describe())
print(pd.Series(private_lr).shape[0])

# %%
#1111 last
print(pd.Series(submit_lr).describe())
print(pd.Series(submit_lr).shape[0])
print('-'*50)
print(pd.Series(private_lr).describe())
print(pd.Series(private_lr).shape[0])

# %%
submit = pd.concat([pd.Series(submit_lr), pd.Series(private_lr)], axis=0)

# %%
submit = np.array(submit).reshape(-1)

# %%
submit #1113

# %%
submit #1111last

# %%
submit_1113 = pd.read_csv('public_private_submission_template.csv')

# %%
submit_1113['predicted_price'] = pd.Series(submit)
submit_1113.to_csv('submit_1113_0.3x_0.7c.csv',index = False, encoding='utf-8')

# %%



