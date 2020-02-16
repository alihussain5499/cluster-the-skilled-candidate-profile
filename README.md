# cluster-the-skilled-candidate-profile
There are almost 400 candidates applied for these role.  Your assignment is to build a Machine Learning model ,   which will help the management to Identify the potential candidates  for both the roles from the given data.

import numpy as np
import pandas as pd

#read data
df=pd.read_csv("D:/mystuff/mavoix_ml_sample_dataset.csv")


df1=df.iloc[0:,0:24]

print(df1)

# Combine Python (out of 3) , R Programming (out of 3) , Deep Learning (out of 3)  for dada science
ds=df1.iloc[0:,2:5] 

ds.describe()

print(ds)


# Combine PHP (out of 3),MySQL (out of 3),HTML (out of 3),CSS (out of 3),JavaScript (out of 3) 
# for web developer	 
 
wd=df1.iloc[0:,6:10]
print(wd)

wd.describe()

# for data scientist

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
X=np.array(ds)
print(X)
X.shape
model.fit(X)

model.cluster_centers_


def dist(a,b):
    return ((a-b)**2).sum()**0.5


centr=model.cluster_centers_
grps=[]
for c in X:
    if dist(c,centr[0]>dist(c,centr[1])):
        grps.append(("select"))
    else:
        grps.append(("reject"))

for g in grps:
    print(g)

len(grps)


data=np.array(df.iloc[0:,0:2])
print(data)

data.shape



data_Scientist_profile=np.c_[data,grps]

print(data_Scientist_profile)

# for web developers

from sklearn.cluster import KMeans
model=KMeans(n_clusters=2,random_state=0)
X1=np.array(wd)
print(X1)
X.shape
model.fit(X1)

model.cluster_centers_


def dist(a,b):
    return ((a-b)**2).sum()**0.5


centr=model.cluster_centers_
grp=[]
for c in X1:
    if dist(c,centr[0]<dist(c,centr[1])):
        grp.append(("select"))
    else:
        grp.append(("reject"))

for g in grp:
    print(g)

data=np.array(df.iloc[0:,0:2])
print(data)

data.shape



web_dev_profile=np.c_[data,grp]

print(web_dev_profile)
