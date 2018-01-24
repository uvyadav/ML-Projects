import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import sklearn.preprocessing as preprocessing
import os
import pandas as pd
import numpy as np
%pylab inline
# Reading training dataset (only those observations where booking is made)
"""file = open("train.csv")
fout = open('subset_datatrain.csv','w')
#n = 0
fout.write(file.readline())
for line in file:
    arr = line.strip().split(',')
    is_book = int(arr[-6])
    if is_book == 1:
        fout.write(line)
fout.close()
file.close()
"""
df = pd.read_csv('subset_datatrain.csv')  # resampling 
X = pd.read_csv("sample_submission.csv")
np.unique(X['hotel_cluster'])

# 3000686 observations (with booking)
#c = np.unique(df['hotel_cluster'])
df2 = pd.read_csv("destinations.csv")
len(np.unique(df['srch_destination_id']))
len(set(df2['srch_destination_id'])-set(df['srch_destination_id']))
len(set(df2['srch_destination_id'])-set(df['srch_destination_id']))

# creating new feature variables for time-stamp data
df['day_of_month'] = pd.to_datetime(df['date_time']).dt.day
df['month'] = pd.to_datetime(df['date_time']).dt.month
df['year'] = pd.to_datetime(df['date_time']).dt.year
# df.columns

#X = df.iloc[:, ].values

del df['orig_destination_distance']
# 62106 rows x 150 columns
# Pre-processing 
# A). Taking care of categorical feature variables
# B). 

#features not considered, for now - date_time, user_location_region, orig_destination_distance, is_mobile, srch_ci, srch_co - no of days of stay
"""features = ['posa_continent','user_id','site_name','user_location_country',
       'user_location_city', 'is_package','channel', 'srch_adults_cnt', 'srch_children_cnt',
       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id','cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
       'hotel_cluster', 'day_of_month', 'month', 'year']"""

features = ['posa_continent','site_name','user_location_country',
       'user_location_city', 'is_package','channel', 'srch_adults_cnt', 'srch_children_cnt',
       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id','cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
       'hotel_cluster', 'day_of_month', 'month', 'year']


df5 = df.loc[:400000,features]
# One hot encoding for categorical features

df6 = df.loc[400001:600000,features] # used to tests
# Performing k-means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
import time
t=time.time()
y_clusters5 = kmeans.fit_predict(df5)
print(time.time()-t)

# Implement an SVM model
t=time.time()
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(df5, y_clusters5)
y_pred5 = classifier.predict(df6, probability = True)
print(time.time()-t)

#implementing for a cluster size of 10,15,20,25
kmeans = KMeans(n_clusters = 10, init = 'k-means++', random_state = 0)
y_clusters10 = kmeans.fit_predict(df5)
classifier10 = SVC(kernel = 'poly', random_state = 0)
classifier10.fit(df5, y_clusters10)
y_pred10 = classifier10.predict(df6, probability = True)


#implementing for a cluster size of 10,15,20,25
kmeans = KMeans(n_clusters = 15, init = 'k-means++', random_state = 0)
y_clusters15 = kmeans.fit_predict(df5)
classifier15 = SVC(kernel = 'poly', random_state = 0)
classifier15.fit(df5, y_clusters15)
y_pred15 = classifier15.predict(df6, probability = True)

#implementing for a cluster size of 10,15,20,25
kmeans = KMeans(n_clusters = 20, init = 'k-means++', random_state = 0)
y_clusters20 = kmeans.fit_predict(df5)
classifier20 = SVC(kernel = 'poly', random_state = 0)
classifier20.fit(df5, y_clusters20)
y_pred20 = classifier20.predict(df6, probability = True)



c0 = len(df5[df5['cluster']==0])
c1 = len(df5[df5['cluster']==1])
c2 = len(df5[df5['cluster']==2])
c3 = len(df5[df5['cluster']==3])
c4 = len(df5[df5['cluster']==4])
index = np.array(range(200001))
df5['indexx']= index
df.groupby(['hotel_cluster', 'cluster']).size().groupby(level=1).max()
df5.groupby(['hotel_cluster','cluster']).size().reset_index().groupby('hotel_cluster')[[0]].max()
pivot_df = df5.pivot(index='hotel_cluster', columns='cluster', values='indexx')


# step - 2 merge the features of destination.csv file
#df3 = pd.merge(df, df2, on='srch_destination_id', how='left')
 
df6 = pd.merge(df5, df2, on='srch_destination_id', how='inner')
# df6 - contains 2988177 x 169 

y = df6['hotel_cluster']  # response variable
del df6['hotel_cluster']
# 168 predictors 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df6, y, test_size = 0.4, random_state = 0)

# Taking care of categorical variables and performing standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import f1_score
score = f1_score(y_test,y_pred,average = 'micro')

