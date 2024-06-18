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








