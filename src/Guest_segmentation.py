# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 23:41:22 2024

@author: dinit
"""

#Import mysql connector
import mysql.connector

#import libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import errors as err
import sklearn.metrics as skmet
from sklearn.preprocessing import MinMaxScaler



#Establish MySQL connection
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='rootpass',
    database='pms_system_bluewaters'
)

#SQL Queries
# expenses and reservations tables
sql_1 ="SELECT booking_no as booking_nu, room_id as room_id,date as date,guest_id as guest_id from reservations";
sql_2 ="SELECT booking_no as booking_nu,guest_id as guest_id, type as type,room_charge as room_charge,net as net_amount,gross as gross_amount, date as date from expences";

#Execute SQL Queries
df_reservations = pd.read_sql(sql_1,db_connection)
df_expenses = pd.read_sql(sql_2,db_connection)

#Remove Duplicates
df_reservations = df_reservations.drop_duplicates()
df_expenses = df_expenses.drop_duplicates()

#Merge resesrvations and expenses dataframes
df_guest_booking = pd.merge(df_reservations, df_expenses, on=['booking_nu', 'guest_id','date'], how='inner')

#Drop missing values
df_guest_booking = df_guest_booking.dropna()

#convert date column to date time format
df_guest_bookings = df_guest_booking.copy()
df_guest_bookings['date'] = pd.to_datetime(df_guest_booking['date'])

#Extract only the guest with type accomodation
df_guest_bookings = df_guest_bookings[df_guest_bookings['type']=='Accommodation']

#current date for Recency Calculation
#current_date = datetime.today()
my_date = datetime(2023, 12, 31)
current_date = my_date

# Identify consecutive days where the same guest stays
df_guest_bookings['previous_date'] = df_guest_bookings.groupby('guest_id')['date'].shift(1)
df_guest_bookings['is_new_booking'] = (df_guest_bookings['date'] - df_guest_bookings['previous_date']).dt.days != 1

#New data frame with only the new bookings - not considering the the same guest stay in a few consecutive days
df_guest_bookings_unique =df_guest_bookings[df_guest_bookings['is_new_booking']==True]

#RFC calculation
#recency - calculate the number of days since the last booking of each guest
#frequency - calculate number of bookings for each guest
#monetory - sum of total amount spent by the guest
df_rfm = df_guest_bookings_unique.groupby('guest_id').agg({
    'date': lambda x: (current_date - x.max()).days,
    'booking_nu': 'count',
    'net_amount': 'sum'
}).rename(columns={'date': 'Recency','booking_nu':'Frequency','net_amount':'net_amount'})

#Draw a scatter plot to show the rfm values of df_rfm dataframe
#x-Recency, Y- Frequency , Size of the datapoint - Net amount
#divide/1000 - make smaller numbers easier to read the graph
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_rfm['Recency'], df_rfm['Frequency'], s=df_rfm['net_amount'] / 1000, alpha=0.5)
plt.title('Scatter Plot of Recency, Frequency, and Net Amount')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#Test number of clusters from 2-10
for n in range(2, 10):
     kmeans = KMeans(n_clusters=n)
     kmeans.fit(df_rfm[['Recency','Frequency','net_amount']])
     labels = kmeans.labels_

     # extract the estimated cluster centres
     cen = kmeans.cluster_centers_
     # calculate the silhoutte score
     print(n, skmet.silhouette_score(
         df_rfm[['Recency','Frequency','net_amount']], labels))

   

#Test k values from 1- 10
k_values = [1,2,3,4,5,6,7,8,9,10] 
#create a list to record wcss_error
wcss_error =[]
#Loop through k values
for k in k_values:
    model=KMeans(n_clusters=k)
    model.fit(df_rfm[['Recency','Frequency','net_amount']])
    wcss_error.append(model.inertia_)
    
