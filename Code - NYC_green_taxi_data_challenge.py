# Importing required libraries
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kstest, ks_2samp, ranksums
from sklearn.preprocessing import normalize, scale
import datetime as dt
from tabulate import tabulate
from datetime import date
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

# setting working directory
import os;
print (os.getcwd())
os.chdir('c:\\Users\uname\desktop\python')

''' ------------------------- Question 1 -----------------------'''
# PART - A :-Downloading the dataset from the website
my_url = "https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2015-09.csv"
dataset = pd.read_csv(my_url)

#for manual loading the dataset-
#dataset = pd.read_csv('green_tripdata_2015-09.csv')
#dataset.info() # Initial visualizing the data

# PART - B :-Printing the number of rows and columns in the dataset
print ("Number of rows:-", dataset.shape[0])
print ("Number of columns:- ", dataset.shape[1])

''' ------------------------- Question 2 -----------------------'''
# PART - A :- Ploting a histogram for "Trip Distance"

fig,ax = plt.subplots(1,2,figsize = (15,4)) 

# histogram of the number of trip distance
dataset.Trip_distance.hist(bins=30,ax=ax[0])
ax[0].set_xlabel('Trip distance (in miles)')
ax[0].set_ylabel('Number')
ax[0].set_yscale('log')
ax[0].set_title('Histogram of trip distance  - with outliers')

# creating a vector to contain Trip Distance
z = dataset.Trip_distance 
# exclude any data point located further than 3 standard deviations of the median point and plotting the histogram with 30 bins
z[~((z-z.median()).abs()>3*z.std())].hist(bins=30,ax=ax[1]) 
ax[1].set_xlabel('Trip distance (in miles)')
ax[1].set_ylabel('Number')
ax[1].set_title('Histogram of trip distance - without outliers')

# applying a lognormal fit and using the mean of trip distance as the scale parameter
scatter,loc,mean = lognorm.fit(dataset.Trip_distance.values,scale=dataset.Trip_distance.mean(),loc=0)
pdf_fitted = lognorm.pdf(np.arange(0,12,.1),scatter,loc,mean)
ax[1].plot(np.arange(0,12,.1),600000*pdf_fitted,'g') 
ax[1].legend(['Lognormal fit','Dataset'])

plt.savefig('Question 2.jpeg',format='jpeg')
plt.show()

# PART - B : - Report any structure you find and any hypotheses you have about that structure
"""
HYPOTHESIS - The distribution of Trip distance is quite asymmetrically with skewness towards the right.  
Since we don't have a symmetric Normal Distribution, it means our dataset (trip distance) is not random. 
The distribution of trip distance has a structure of lognormal distribution
"""

''' ------------------------- Question 3 -----------------------'''
# PART - A :- Report mean and median trip distance grouped by hour of day
# Making new feature variable for pickup and drop off datetime in the specific right format
dataset['Pickup_dt'] = dataset.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
dataset['Dropoff_dt'] = dataset.Lpep_dropoff_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

# Creating a feature variable for pickup hours (0-23)
dataset['Pickup_hour'] = dataset.Pickup_dt.apply(lambda x:x.hour)

# Mean and Median of trip distance by pickup hour
fig,ax = plt.subplots(1,1,figsize=(9,5)) # prepare fig to plot mean and median values
# use a pivot table to aggregate Trip_distance by hour
table1 = dataset.pivot_table(index='Pickup_hour', values='Trip_distance',aggfunc=('mean','median')).reset_index()
table1.columns = ['Hour','Mean_distance','Median_distance']
table1[['Mean_distance','Median_distance']].plot(ax=ax)
plt.ylabel('Trip distance (in miles)')
plt.xlabel('Hours')
plt.title('Distribution of trip distance by pickup hour')
plt.xlim([0,23])
plt.savefig('Question 3_1.jpeg',format='jpeg')
plt.show()
print ('-----Trip distance by hour of the day-----\n')
print (tabulate(table1.values.tolist(),["Hour","Mean distance","Median distance"]))

# PART - B :- Identifying trips that originate or terminate at one of the NYC area airports

# selecting airport trips
airports_trips = dataset[(dataset.RateCodeID==2) | (dataset.RateCodeID==3)]
print ("Number of trips to/from NYC airports: ", airports_trips.shape[0])
print ("Average fare of trips to/from NYC airports: $", airports_trips.Fare_amount.mean(),"per trip")
print ("Average total charged amount (before tip) of trips to/from NYC airports: $", airports_trips.Total_amount.mean(),"per trip")

# creating a vector to contain Trip Distance 
z2 = airports_trips.Trip_distance # airport trips
z3 = dataset.loc[~dataset.index.isin(z2.index),'Trip_distance'] # non-airport trips

# removing outliers: excluding any data point located further than 3 standard deviations of the median point and plotting the histogram with 30 bins
z2 = z2[~((z2-z2.median()).abs()>3*z2.std())]
z3 = z3[~((z3-z3.median()).abs()>3*z3.std())] 

# defining the bins boundaries
bins = np.histogram(z2,normed=True)[1]
h2 = np.histogram(z2,bins=bins,normed=True)
h3 = np.histogram(z3,bins=bins,normed=True)

# plotting distributions of trip distance normalized among groups
fig,ax = plt.subplots(1,2,figsize = (15,4))
w = .4*(bins[1]-bins[0])
ax[0].bar(bins[:-1],h2[0],alpha=1,width=w,color='b')
ax[0].bar(bins[:-1]+w,h3[0],alpha=1,width=w,color='g')
ax[0].legend(['Airport trips','Non-airport trips'],loc='best',title='Total')
ax[0].set_xlabel('Trip distance (in miles)')
ax[0].set_ylabel('Group normalized trips count')
ax[0].set_title('A. Trip distance distribution')

# plotting hourly distribution
airports_trips.Pickup_hour.value_counts(normalize=True).sort_index().plot(ax=ax[1])
dataset.loc[~dataset.index.isin(z2.index),'Pickup_hour'].value_counts(normalize=True).sort_index().plot(ax=ax[1])
ax[1].set_xlabel('Hours')
ax[1].set_ylabel('Group normalized trips count')
ax[1].set_title('B. Hourly distribution of trips')
ax[1].legend(['Airport trips','Non-airport trips'],loc='best',title='Total')
plt.savefig('Question 3_2.jpeg',format='jpeg')
plt.show()

''' ------------------------- Question 4 -----------------------'''
# PART - A:- Building a derived variable for tip as a percentage of total fare
dataset = dataset[(dataset.Total_amount>=2.5)] # Since the min. charge of NYC green taxi is $2.5
dataset['Tip_percentage'] = 100*dataset.Tip_amount/dataset.Total_amount
print ("Summary: Tip percentage\n",dataset.Tip_percentage.describe())


# PART - B :- Prediction model for tip_percentage variable

# Data Cleaning

dataset.drop('Ehail_fee',axis=1,inplace=True) # Removing the column - Ehail_fee as it contains NaN
dataset['Trip_type '] = dataset['Trip_type '].replace(np.NaN,1)
# Since amount can't be -ve, so cleaning all the related -ve data
dataset.Total_amount= dataset.Total_amount.abs()
dataset.Fare_amount = dataset.Fare_amount.abs()
dataset.improvement_surcharge = dataset.improvement_surcharge.abs()
dataset.Tip_amount = dataset.Tip_amount.abs()
dataset.Tolls_amount = dataset.Tolls_amount.abs()
dataset.MTA_tax = dataset.MTA_tax.abs()

# Checking for NaN in the dataset
a = np.argwhere(np.isnan(dataset['Total_amount']))
b = np.argwhere(np.isnan(dataset['Trip_distance']))
c = np.argwhere(np.isnan(dataset['Payment_type']))
d = np.argwhere(np.isnan(dataset['Avg_speed']))

# Exploratory data analysis

# Creating new feature variables for days, weeks, hours
#Calculating avg. speed for each trip
pickup = pd.DataFrame(dataset.lpep_pickup_datetime.str.split(' ',1).tolist(),columns = ['Pickup_Date','Pickup_Time'])
dataset['Pickup_date']= pickup['Pickup_Date']
drop = pd.DataFrame(dataset.Lpep_dropoff_datetime.str.split(' ',1).tolist(),columns = ['Drop_Date','Drop_Time'])
dataset['Drop_date']= drop['Drop_Date']
dataset['pickuptime'] = pd.to_datetime(dataset.lpep_pickup_datetime)
dataset['droptime'] = pd.to_datetime(dataset.Lpep_dropoff_datetime)
dataset['traveltime'] = dataset['droptime'] - dataset['pickuptime']
dataset['traveltime_hrs'] = dataset['traveltime'] / np.timedelta64(1, 'h')
dataset['Avg_speed'] = dataset['Trip_distance']/dataset['traveltime_hrs'] 
dataset['Pickup_dt'] = dataset.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
dataset['Dropoff_dt'] = dataset.lpep_pickup_datetime.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
ref_week = dt.datetime(2015,9,1).isocalendar()[1] 
dataset['Week'] = dataset.Pickup_dt.apply(lambda x:x.isocalendar()[1])-ref_week+1
dataset['Week_day']  = dataset.Pickup_dt.apply(lambda x:x.isocalendar()[2])
dataset['Month_day'] = dataset.Pickup_dt.apply(lambda x:x.day)
dataset['Hour'] = dataset.Pickup_dt.apply(lambda x:x.hour)
dataset['Day_of_week'] = pd.to_datetime(dataset['Pickup_date']).dt.weekday_name
dataset['Trip_duration'] = ((dataset.Dropoff_dt-dataset.Pickup_dt).apply(lambda x:x.total_seconds()/60.))

# Avg_speed in miles per hrs 
dataset.loc[np.isnan(dataset.Avg_speed), 'Avg_speed'] = 30 # substituting outliers at the end 

# Creating histograms for tip_percentage 
fig,ax=plt.subplots(1,2,figsize=(14,4))
dataset.Tip_percentage.hist(bins = 20,normed=True,ax=ax[0])
ax[0].set_xlabel('Tip (%)')
ax[0].set_title('Distribution of Tip (%) - All transactions')

dataset1 = dataset[dataset.Tip_percentage>0]
dataset1.Tip_percentage.hist(bins = 20,normed=True,ax=ax[1])
ax[1].set_xlabel('Tip (%)')
ax[1].set_title('Distribution of Tip (%) - Transaction with tips')
ax[1].set_ylabel('Group normed count')
plt.savefig('Question 4_a.jpeg',format='jpeg')
plt.show()

#Building a predictive model using RandomForest 

rf = RandomForestRegressor(6, )

#Considering the feature variables for the predictive model
predictors = ['Total_amount', 'Trip_duration', 'Avg_speed', 'Passenger_count',  
              'RateCodeID', 'Payment_type', 'Pickup_longitude', 
       'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude' ]

dataX = dataset[predictors]
dataY = dataset['Tip_percentage']

# spliting the dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 0.3, random_state = 0)

rf.fit(X_train, y_train)
r_score = rf.score(X_test, y_test)
print ("R-square for the prediction model built using Randomforest : ", r_score)

# OBSERVATION - Mean accuracy score obtained from the test-data is 0.9216, which is quite significant


''' ------------------------- Question 5 -----------------------'''
# I will perform a clustering based on geographic location data. Will create the NY region in 6 different clusters
# Then to analyze the traffic behaviour, I will use parameter - avg_speed of the trips. 
         
# Applying DBSCAN clustering algorithm
#from scipy.spatial.distance import pdist, squareform
#from sklearn.cluster import DBSCAN
#db = DBSCAN(eps=.2,min_samples=10)
#X = dataset.iloc[0:10000,5:7]
#y_db = db.fit_predict(X)
#set(y_db)
# It's been observed that DBSCAN is consuming too much time for geographic clustering of data
# Although DBSCAN is far better optimized algorithm as compared to K-means clustering, 
#but due to time limitation we will go with the K-means clustering 

# Fitting K-Means clustering algo to the dataset
from sklearn.cluster import KMeans

X = dataset.iloc[:,5:7].values

wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 6), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

from scipy.stats import itemfreq
itemfreq(y_kmeans)
    
# Making cluster as a new feature variable
dataset['Cluster']=y_kmeans
dataset1= dataset.iloc[y_kmeans!=1,:]
X = dataset1.iloc[:,5:7].values
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
#plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
#plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 10, c = 'yellow', label = 'Cluster 6')
plt.title('Clusters of customers')
plt.xlabel('Lat')
plt.ylabel('Long')
plt.legend()
plt.savefig('Clusters.jpeg',format='jpeg')
plt.show()

# HC algorithm
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#Calculating avg. speed for each trip since in the previously calculated avg_speed, outliers were replaced with a higher end value, 
# and now we want to actually observe the whole distribtution it wont be a good idea to replace outlier with a common statistic (like mean or median)
dataset['pickuptime'] = pd.to_datetime(dataset.lpep_pickup_datetime)
dataset['droptime'] = pd.to_datetime(dataset.Lpep_dropoff_datetime)
dataset['traveltime'] = dataset['droptime'] - dataset['pickuptime']
dataset['traveltime_hrs'] = dataset['traveltime'] / np.timedelta64(1, 'h')
dataset['Avg_speed'] = dataset['Trip_distance']/dataset['traveltime_hrs'] 
# Avg_speed in miles per hrs 

#dataset['Trip_duration'] = ((dataset.Dropoff_dt-dataset.Pickup_dt).apply(lambda x:x.total_seconds()/60.))
#dataset['Speed_mph'] = dataset.Trip_distance/(dataset.Trip_duration/60)

# Cleaning avg_speed
dataset["Avg_speed"]= dataset["Avg_speed"].replace([np.inf], np.nan) # Taking off trips with zero travel time
dataset.Avg_speed[dataset.Avg_speed > 50].count() # Just checking the outliers in the avg_speed. Although the speed limit for taxi in NY is 25 miles/hr
dataset.loc[dataset.Avg_speed> 50, 'Avg_speed'] = np.nan # Cleaning the outliers 
#dataset['Avg_speed']= dataset['Avg_speed'].replace(dataset.Avg_speed[dataset.Avg_speed] > 50, np.nan) 
#dataset.traveltime_hrs[dataset.traveltime_hrs <0.0167].count()  # counting the trip which took less than 1 min 
#dataset['traveltime_hrs'] = dataset.traveltime_hrs.replace(dataset.traveltime_hrs < 0.0167,0)

# Creating columns for pickup date and drop date
pickup = pd.DataFrame(dataset.lpep_pickup_datetime.str.split(' ',1).tolist(),columns = ['Pickup_Date','Pickup_Time'])
dataset['Pickup_date']= pickup['Pickup_Date']
drop = pd.DataFrame(dataset.Lpep_dropoff_datetime.str.split(' ',1).tolist(),columns = ['Drop_Date','Drop_Time'])
dataset['Drop_date']= drop['Drop_Date']

# Ploting avg_speed by weekdays
dataset[['Avg_speed','Day_of_week']].groupby('Day_of_week').mean()
dataset.boxplot('Avg_speed','Day_of_week')
plt.ylim([2,22]) # cutting off outliers
plt.ylabel('Avg. Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed on different weekdays',fontsize=12,loc='Center')
plt.savefig('Question 5_1.jpeg',format='jpeg')
plt.title('')
plt.show()

# OBSERVATION  - It can be observed that on Sunday, Monday and Saturday the average speeds are usually higher as compared to other week days
# Higher avg speed implies lower traffic on Saturday, Sunday and Mondays on an average

# Comparision of Mean speed in different weeks
dataset.boxplot('Avg_speed','Week')
plt.ylim([0,22]) # cutting off the outliers
plt.ylabel('Average Speed (in miles/hr) ')
plt.suptitle('')
plt.title('Distribution of avg speed in different weeks',fontsize=12,loc='Center')
plt.savefig('Question 5_2.jpeg',format='jpeg')
plt.title('')
plt.show()
# Plotting the Mean speed distribution in different weeks
x1 = dataset.loc[dataset.Week==1,'Avg_speed']
x2 = dataset.loc[dataset.Week==2,'Avg_speed']
x3 = dataset.loc[dataset.Week==3,'Avg_speed']
x4 = dataset.loc[dataset.Week==4,'Avg_speed']
x5 = dataset.loc[dataset.Week==5,'Avg_speed']

a = np.argwhere(np.isnan(x1)) # to check NaN entries
x1 = x1[np.logical_not(np.isnan(x1))]
x2 = x2[np.logical_not(np.isnan(x2))]
x3 = x3[np.logical_not(np.isnan(x3))]
x4 = x4[np.logical_not(np.isnan(x4))]
x5 = x5[np.logical_not(np.isnan(x5))]

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
plt.hist(x1, normed=False, bins=30)
plt.xlabel('Avg. Speed (in miles per hr)')
plt.ylabel('Count')
plt.title('Distribution of avg speed in Week - 1')
plt.savefig('Question 5_3a.jpeg',format='jpeg')
plt.show()

plt.hist(x2, normed=False, bins=30)
plt.xlabel('Avg. Speed (in miles per hr)')
plt.ylabel('Count')
plt.title('Distribution of avg speed in Week - 2')
plt.savefig('Question 5_3b.jpeg',format='jpeg')
plt.show()

plt.hist(x3, normed=False, bins=30)
plt.xlabel('Avg. Speed (in miles per hr)')
plt.ylabel('Count')
plt.title('Distribution of avg speed in Week - 3')
plt.savefig('Question 5_3c.jpeg',format='jpeg')
plt.show()

plt.hist(x4, normed=False, bins=30)
plt.xlabel('Avg. Speed (in miles per hr)')
plt.ylabel('Count')
plt.title('Distribution of avg speed in Week - 4')
plt.savefig('Question 5_3d.jpeg',format='jpeg')
plt.show()

plt.hist(x5, normed=False, bins=30)
plt.xlabel('Avg. Speed (in miles per hr)')
plt.ylabel('Count')
plt.title('Distribution of avg speed in Week - 5')
plt.savefig('Question 5_3e.jpeg',format='jpeg')
plt.show()

# OBSERVATION - It can be observed that the distribution of avg speed in all the five weeks are approx. similar
# For checking this hypothesis, various tests can be performed - Kolmogrov-Smirnov Test, Anderson-Darling Test, Wilcoxon rank-sum test

# Performing Kolmogrov - Smirnov test to compare the distributions of Avg_speed across diff. weeks in Sep
from scipy.stats import kstest, ks_2samp
ks_2samp(x1,x3)
ks_2samp(x2,x3)
ks_2samp(x3,x4)
ks_2samp(x2,x5) # can also be checked for all the possible combinations 

# OBSERVATIONS - Its been observed that K-S statistic is found to be very small (less than 0.05) in all the cases.
# Also one can note that the p-value resulted from the KS-test is also quite small, which can't be trusted here with a dataset so big

# Plotting avg_speed against hours
dataset[['Avg_speed','Pickup_hour']].groupby('Pickup_hour').mean()
dataset.boxplot('Avg_speed','Pickup_hour')
plt.ylim([2,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed w.r.t. Pickup_hours',fontsize=12,loc='Center')
plt.savefig('Question 5_4.jpeg',format='jpeg')
plt.title('')
plt.show()
# OBSERVATION - From the boxplot, it can be observed that the traffic is faster in early morning timings and gets slower in the evenings

# Plotting avg_speed in different clusters
dataset[['Avg_speed','Cluster']].groupby('Cluster').mean()
dataset.boxplot('Avg_speed','Cluster')
plt.ylim([1,25]) # cut off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in diff. Clusters',fontsize=12,loc='Center')
plt.savefig('Question 5_5.jpeg',format='jpeg')
plt.title('')
plt.show()
#dataset.loc[dataset.Cluster==1,'Avg_speed']

# OBSERVATION - Avg_speed is comparatively very low in Cluster-1 when compared 
#with other cluster regions, may indicate heavy traffic in that area
# I will be checking this thing in more details later, how the traffic is varying daywise in Cluster-1 in Sep month

# Plotting avg_speed on particular days of month
dataset[['Avg_speed','Month_day']].groupby('Month_day').mean()
dataset.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.savefig('Question 5_6.jpeg',format='jpeg')
plt.title('')
plt.show()

# Plotting avg_speed on a particular day in a particular cluster
dataset = dataset.reset_index(drop=True)
# Cluster- 0
c0=dataset.loc[dataset.Cluster==0,('Avg_speed','Month_day')]
c0[['Avg_speed','Month_day']].groupby('Month_day').mean()
c0.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-0',fontsize=12,loc='Center')
plt.savefig('Question 5_7a.jpeg',format='jpeg')
plt.show()
# Cluster-1
c1=dataset.loc[dataset.Cluster==1,('Avg_speed','Month_day')]
c1[['Avg_speed','Month_day']].groupby('Month_day').mean()
c1.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-1',fontsize=12,loc='Center')
plt.savefig('Question 5_7b.jpeg',format='jpeg')
plt.show()
# Cluster-2
c2=dataset.loc[dataset.Cluster==2,('Avg_speed','Month_day')]
c2[['Avg_speed','Month_day']].groupby('Month_day').mean()
c2.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-2',fontsize=12,loc='Center')
plt.savefig('Question 5_7c.jpeg',format='jpeg')
plt.show()
# Cluster-3
c3=dataset.loc[dataset.Cluster==3,('Avg_speed','Month_day')]
c3[['Avg_speed','Month_day']].groupby('Month_day').mean()
c3.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-3',fontsize=12,loc='Center')
plt.savefig('Question 5_7d.jpeg',format='jpeg')
plt.show()
# Cluster-4
c4=dataset.loc[dataset.Cluster==4,('Avg_speed','Month_day')]
c4[['Avg_speed','Month_day']].groupby('Month_day').mean()
c4.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-4',fontsize=12,loc='Center')
plt.savefig('Question 5_7e.jpeg',format='jpeg')
plt.show()
# Cluster-5
c5=dataset.loc[dataset.Cluster==5,('Avg_speed','Month_day')]
c5[['Avg_speed','Month_day']].groupby('Month_day').mean()
c5.boxplot('Avg_speed','Month_day')
plt.ylim([1,22]) # cutting off outliers
plt.ylabel('Speed (in miles per hr)')
plt.suptitle('')
plt.title('Distribution of avg speed in Cluster-5',fontsize=12,loc='Center')
plt.savefig('Question 5_7f.jpeg',format='jpeg')
plt.show()

# OBSERVATIONS - The avg speed touched 2.5mph in cluster-1 on Sep-17, indicates huge traffic in that area on that particular day




