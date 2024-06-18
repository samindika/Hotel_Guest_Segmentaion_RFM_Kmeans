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