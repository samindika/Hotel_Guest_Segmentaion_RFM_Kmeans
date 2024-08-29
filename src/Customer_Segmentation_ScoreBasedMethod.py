
# Data Preprocessing, Mannual outlier detection, Direct K-means clustering , Use of Demographic variables

# Import MySQL connector
import mysql.connector

# Import Libraries for data manipulation(pandas), data visualization (matplotlib,seaborn) and numerical operations(numpy)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import libraries for machine learning (KMeans),error handling, metrics, data scaling and statistical functions
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew

# Import datetime module from the datetime library
from datetime import datetime

# Import the silhouette_score function from sklearn.metrics for evaluating the quality of clustering
from sklearn.metrics import silhouette_score

# Suppress specific warnings related to SQLAlchemy connectable support in pandas
import warnings
warnings.filterwarnings(
    "ignore", message="pandas only support SQLAlchemy connectable")

# Establishing connection to MySQL database
db_connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='rootpass',
    database='pms_system_bluewaters'
)

# SQL Queries to select guest information and booking details
sql_1 = "SELECT id as guest_id, gender as gender, country_name as country from new_guest_details"
sql_2 = "SELECT booking_no as booking_nu,guest_id as guest_id, room_id as room_id, type as type,room_charge as room_charge,net as net_amount,gross as gross_amount, date as date, status as status from expences"

# Execute SQL queries to load data into pandas dataframes from MySQL database connection
df_guests = pd.read_sql(sql_1, db_connection)
df_expenses = pd.read_sql(sql_2, db_connection)

# Filter the dataframe to include only rows where status = 1, type= Accomodatoin, Year= 2018
# Convert date into date time, invalid date will be set as NaT
df_expenses = df_expenses[df_expenses['status'] == 1]
df_expenses = df_expenses[df_expenses['type'] == 'Accommodation']
df_guest_bookings = df_expenses.copy()
df_guest_bookings['date'] = pd.to_datetime(
    df_guest_bookings['date'], errors='coerce')
df_guest_bookings = df_guest_bookings[df_guest_bookings['date'].dt.year.isin([
                                                                             2018])]

# Check for missing values in the guest booking dataframe
print(df_guest_bookings.isnull().any())

# Display number of missing values in the guest booking dataframe
print(df_guest_bookings.isnull().sum())

# Replace missing values in the guest_id column with mean value
df_guest_bookings['guest_id'].fillna(
    (df_guest_bookings['guest_id'].mean()), inplace=True)

# Checking duplicate records in the guest booking dataframe
print(df_guest_bookings.duplicated().sum())

# Display summary of guest booking dataframe
print(df_guest_bookings.info())

# Display summary statistics for numerical columns in the guest booking dataframe
print(df_guest_bookings.describe())

# Check for missing values in the guest dataframe
print(df_guests.isnull().any())

# Display number of missing values in the guest dataframe
print(df_guests.isnull().sum())

# Replace all missing values (gender, country) with the string 'Unknown' in the guest dataframe
df_guests.fillna('unknown', inplace=True)

# Checking duplicate records in the guest dataframe
print(df_guests.duplicated().sum())

# Count the number of occurances of each unique value in the gender column
df_guestgender_counts = df_guests['gender'].value_counts()
print(df_guestgender_counts)

# Count the number of occurances of each unique value in the country column
df_guestcountry_counts = df_guests['country'].value_counts()
print(df_guestcountry_counts)

# Convert guest_id column in the df_guests data frame into float
df_guests['guest_id'] = df_guests['guest_id'].astype(float)

# Merge df_guest_bookings and df_guests on  guest_id using left join to preserve all rows in df_guest_bookings dataframe
df_guest_bookings = pd.merge(
    df_guest_bookings, df_guests, on='guest_id', how='left')
print(df_guest_bookings)

# Check for missing values in the merged dataframe
print(df_guest_bookings.isnull().any())

# Display number of missing values in the merged dataframe
print(df_guest_bookings.isnull().sum())

# Checking duplicate records in the merged dataframe
print(df_guest_bookings.duplicated().sum())

# Replace all missing values of gender and country columns with the string 'Unknown'
df_guest_bookings['gender'].fillna('unknown', inplace=True)
df_guest_bookings['country'].fillna('unknown', inplace=True)

# Identify consecutive days where the same guest stays
df_guest_bookings_n = df_guest_bookings.copy()
df_guest_bookings_n['previous_date'] = df_guest_bookings_n.groupby('guest_id')[
    'date'].shift(1)
df_guest_bookings_n['is_new_booking'] = (
    df_guest_bookings_n['date'] - df_guest_bookings_n['previous_date']).dt.days != 1
print(df_guest_bookings_n)

# New data frame with only the new bookings - not consider the the same guest stay in a few consecutive days
df_guest_bookings_unique = df_guest_bookings_n[df_guest_bookings_n['is_new_booking'] == True]
print(df_guest_bookings_unique)

# Select specific columns in the dataframe
df_guest_bookings_unique = df_guest_bookings_unique[[
    'guest_id', 'room_id', 'booking_nu', 'net_amount', 'gross_amount', 'date', 'gender', 'country', 'status']]
print(df_guest_bookings_unique)

# Count the number of occurrences of each unique value in the 'gender' column of the merged dataframe
df_guest_counts = df_guest_bookings_unique['gender'].value_counts()

# ---------------Plot the gender distribution of the dataset --------------------------------
labels = df_guest_counts.index
colors = sns.color_palette('pastel')[0:len(df_guest_counts)]

# Calculate percentages
percentages = 100 * df_guest_counts / df_guest_counts.sum()
legend_labels = [f'{label}: {percentage:.3f}%' for label,
                 percentage in zip(labels, percentages)]

# Plot the pie chart without labels on the slices
plt.figure(figsize=(10, 7))
plt.pie(df_guest_counts, colors=colors,
        startangle=140, textprops=dict(color="w"))

# Add a legend with the labels and percentages
plt.legend(legend_labels, title="Gender",
           loc='center left', bbox_to_anchor=(1, 0.5))

plt.title('Gender Distribution of Guests')
plt.show()

# Count the number of occurrences of each unique value in the 'country' column of merged the dataframe
df_guest_country = df_guest_bookings_unique['country'].value_counts()

# ---------------Plot the country distribution of the dataset --------------------------------
labels = df_guest_country.index
colors = sns.color_palette('pastel')[0:len(df_guest_country)]

# Calculate percentages
percentages = 100 * df_guest_country / df_guest_country.sum()
legend_labels = [f'{label}: {percentage:.3f}%' for label,
                 percentage in zip(labels, percentages)]

# Plot the pie chart without labels on the slices
plt.figure(figsize=(10, 7))
plt.pie(df_guest_country, colors=colors,
        startangle=140, textprops=dict(color="w"))

# Add a legend with the labels and percentages
plt.legend(legend_labels, title="Countries",
           loc='center left', bbox_to_anchor=(1, 0.5))

plt.title('Country Distribution of Guests')
plt.show()

# Summary statistics for numerical columns in the merged dataframe
print(df_guest_bookings_unique.describe())

# Count the number of guest bookings with net amount = 0
zero_net_amount_count = df_guest_bookings_unique[df_guest_bookings_unique['net_amount'] == 0].shape[0]
print(zero_net_amount_count)

# Calculate the percentage of rows in df_guest_bookings_unique where 'net_amount' is 0
percentage = zero_net_amount_count / \
    df_guest_bookings_unique['net_amount'].count() * 100
print(percentage)

# Filter out rows from df_guest_bookings_unique where 'net_amount' is 0
df_guest_bookings_unique = df_guest_bookings_unique[df_guest_bookings_unique['net_amount'] != 0]

# Count the number of guest bookings with net amount = 0.01
zero_net_amount_count1 = df_guest_bookings_unique[df_guest_bookings_unique['net_amount'] == 0.01].shape[0]

# Calculate the percentage of rows in df_guest_bookings_unique where 'net_amount' is 0.01
percentage = zero_net_amount_count1 / \
    df_guest_bookings_unique['net_amount'].count() * 100

# Filter out rows from df_guest_bookings_unique where 'net_amount' is not equal to 0.01
df_guest_bookings_unique = df_guest_bookings_unique[df_guest_bookings_unique['net_amount'] != 0.01]

# Filter and display rows from df_guest_bookings_unique where the 'date' column contains missing (NaN) values
# No garbage values for date
garbage_dates = df_guest_bookings_unique[df_guest_bookings_unique['date'].isna(
)]
print(garbage_dates)

# ---------------------KDE and Box plots of Net Amount -------------------

# Plot 1: Kernel Density Estimate (KDE) plot for 'net_amount'
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
sns.kdeplot(x=df_guest_bookings_unique['net_amount'], bw_adjust=0.5)
plt.title('Kernel Density Plot of Net amount')
plt.xlabel('Net amount')
plt.ylabel('Density')

# Plot 2: Box plot for net amount
plt.subplot(1, 3, 2)
sns.boxplot(x=df_guest_bookings_unique['net_amount'])
plt.title('Box plot of Net amount')
plt.xlabel('Net amount')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Calculate the skweness of the net amount
#Skewness > 0 --positive
#Skewness <0 --negative
#Skewness = 0 --Symmetric
# 0.5 - 1 ---moderately positive skewed
# Skeweness >1  --Severe positive skeweness
skewness_1 = skew(df_guest_bookings_unique['net_amount'])
print(skewness_1)

# Calculate Q1 and Q3 of net amount
percentile25 = df_guest_bookings_unique['net_amount'].quantile(0.25)
percentile75 = df_guest_bookings_unique['net_amount'].quantile(0.75)
# Use percentiles to compute the Interquartile Range (IQR)
iqr = percentile75 - percentile25
# Determine the upper and lower bounds for detecting outliers using 1.5 times the IQR
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print(upper_limit, lower_limit)

# Count the number of outliers in the dataframe exceeds the upper limit. No values exceed the lower limit
outliers = df_guest_bookings_unique[df_guest_bookings_unique['net_amount']
                                    > upper_limit].shape[0]

# Calculate the percentage of outliers in the df_guest_bookings_unique DataFrame
percentage = outliers / df_guest_bookings_unique['net_amount'].count() * 100

# Use capping method
new_df_cap = df_guest_bookings_unique.copy()
new_df_cap['net_amount'] = np.where(
    new_df_cap['net_amount'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['net_amount'] < lower_limit,
        lower_limit,
        new_df_cap['net_amount']))

# -----------------------KDE and Box plots after capping method------------------------

# Compare plots after capping
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
sns.kdeplot(x=new_df_cap['net_amount'], bw_adjust=0.5)
plt.title('Kernel Density Plot of Net Amount')
plt.xlabel('Net Amount')
plt.ylabel('Density')

plt.subplot(1, 3, 2)
sns.boxplot(x=new_df_cap['net_amount'])
plt.title('Box plot of Net Amount')
plt.xlabel('Net Amount')
plt.ylabel('Density')

plt.tight_layout()
plt.show()


# Calculate the skweness of the net amount
skewness = skew(new_df_cap['net_amount'])
print(skewness)

# current date for Recency Calculation
#current_date = datetime.today()
my_date = datetime(2019, 1, 1)
print(my_date)
current_date = my_date

# Aggregate the new_df_cap DataFrame to compute RFM (Recency, Frequency, Monetary) metrics for each guest
# first' to take the first occurrence of the gender associated with each guest_id.
df_rfm = new_df_cap.groupby('guest_id').agg({
    'date': lambda x: (current_date - x.max()).days,
    'booking_nu': 'count',
    'net_amount': 'sum',
    'gender': 'first',
    'country': 'first'
}).reset_index()
df_rfm.columns = ['guest_id', 'Recency',
                  'Frequency', 'Monetary', 'gender', 'country']

# -----------------------KDE and Box plots for RFM values--------------------

# KDE and box plots for Recency, Frequency and Monetary columns
plt.figure(figsize=(20, 10))
plt.subplot(3, 2, 1)
sns.kdeplot(x=df_rfm['Recency'], bw_adjust=0.5)
plt.title('Kernel Density Plot of Recency')
plt.xlabel('Recency')
plt.ylabel('Density')

plt.subplot(3, 2, 2)
sns.boxplot(x=df_rfm['Recency'])
plt.title('Box plot of Recency')
plt.xlabel('Recency')
plt.ylabel('Density')

plt.subplot(3, 2, 3)
sns.kdeplot(x=df_rfm['Frequency'], bw_adjust=0.5)
plt.title('Kernel Density Plot of Frequency')
plt.xlabel('Frequency')
plt.ylabel('Density')

plt.subplot(3, 2, 4)
sns.boxplot(x=df_rfm['Frequency'])
plt.title('Box plot of Frequency')
plt.xlabel('Frequency')
plt.ylabel('Density')


plt.subplot(3, 2, 5)
sns.kdeplot(x=df_rfm['Monetary'], bw_adjust=0.5)
plt.title('Kernel Density Plot of Monetary')
plt.xlabel('Monetary')
plt.ylabel('Density')

plt.subplot(3, 2, 6)
sns.boxplot(x=df_rfm['Monetary'])
plt.title('Box plot of Monetary')
plt.xlabel('Monetary')
plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Calculate skewness
skewness_2 = skew(df_rfm.iloc[:, 1:4])
print(skewness_2)

# Mannualy handle the outliers to avoid rows with high frequency values
df_rfm = df_rfm[df_rfm['Frequency'] < 100]

# Calculate skewness (Skewness is sigificantly reduced)
skewness_3 = skew(df_rfm.iloc[:, 1:4])
print(skewness_3)

# take a new copy of the df_rfm dataframe
df_rfm_new = df_rfm.copy()


# Calculate quantiles for each metric (Recency, Frequency, Monetary)
quantiles = df_rfm_new.quantile(q=[0.25, 0.5, 0.75])

# Function to assign R score based on Recency


def r_score(x):
    if x <= quantiles['Recency'][0.25]:
        return 4
    elif x <= quantiles['Recency'][0.50]:
        return 3
    elif x <= quantiles['Recency'][0.75]:
        return 2
    else:
        return 1


def fm_score(x, col):
    if x <= quantiles[col][0.25]:
        return 1
    elif x <= quantiles[col][0.50]:
        return 2
    elif x <= quantiles[col][0.75]:
        return 3
    else:
        return 4


# Apply the function to assign scores

df_rfm_new.loc[:, 'R_Score'] = df_rfm_new['Recency'].apply(r_score)
df_rfm_new.loc[:, 'F_Score'] = df_rfm_new['Frequency'].apply(
    fm_score, col='Frequency')
df_rfm_new.loc[:, 'M_Score'] = df_rfm_new['Monetary'].apply(
    fm_score, col='Monetary')

# Calculate the overall RFM score
df_rfm_new.loc[:, 'RFM_score'] = df_rfm_new[[
    'R_Score', 'F_Score', 'M_Score']].sum(axis=1)

# Filter necessary columns
df_rfm_score = df_rfm_new[['guest_id', 'R_Score', 'F_Score', 'M_Score']]

# Fit the min max scaler to the RFM scores and transform the data
scaler_minmax = MinMaxScaler()
df_rfm_minmax = scaler_minmax.fit_transform(
    df_rfm_score[['R_Score', 'F_Score', 'M_Score']])
df_rfm_minmax = pd.DataFrame(df_rfm_minmax, columns=[
                             'Recency', 'Frequency', 'Monetory'])

# ---------------------Elbow method to find the optimal number of clusters---------------------
SSE = []
k_range = range(2, 11)  # Trying different number of clusters from 2 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1231)
    kmeans.fit(df_rfm_minmax)
    SSE.append(kmeans.inertia_)

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(k_range), y=SSE, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Perform K-means clustering
model = KMeans(n_clusters=4)
labels = model.fit_predict(df_rfm_minmax[['Recency', 'Frequency', 'Monetory']])
centers = model.cluster_centers_
df_rfm_minmax['cluster'] = labels

# -------------------Create a 3D scatter plot to visualize clusters---------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_rfm_minmax['Recency'], df_rfm_minmax['Frequency'],
           df_rfm_minmax['Monetory'], cmap='brg', c=df_rfm_minmax['cluster'])
ax.scatter(centers[:, 0], centers[:, 1],
           centers[:, 2], s=300, c='black', marker='X')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetory')
ax.set_title('3D Scatter Plot of Clusters with Centers')
plt.show()

# Calculate the silhouette score to evaluate the quality of clustering
# Silhouette score should be close to 1 and low inertia value
silhout_score = silhouette_score(
    df_rfm_minmax[['Recency', 'Frequency', 'Monetory']], labels)
print(silhout_score)


# Align the indices of df_rfm_minmax with those of df_rfm_new
# Assign the cluster labels from df_rfm_minmax to the original dataframe df_rfm_new
df_rfm_new = df_rfm.copy()
df_rfm_minmax.index = df_rfm_new.index
df_rfm_new['cluster'] = df_rfm_minmax['cluster']

# Count the number of data points in each cluster and display the result
print(df_rfm_new['cluster'].value_counts())
