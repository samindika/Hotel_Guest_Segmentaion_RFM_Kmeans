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
