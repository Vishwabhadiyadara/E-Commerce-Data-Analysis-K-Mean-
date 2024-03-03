#!/usr/bin/env python
# coding: utf-8

# # Project 2:- Customer Segmentation using RFM Analysis

# ### Task 1:- Data Preprocessing

# #### Load the dataset:

# In[2]:


import pandas as pd

# Load the dataset into a pandas DataFrame with specified encoding
df = pd.read_csv("C:/Users/vrajd/Downloads/archive/data.csv", encoding='latin1')
df


# #### Display the initial structure of the dataset:

# In[3]:


# Display the initial structure of the dataset
print("Initial Structure:\n")
df.info()


# #### Check for missing values:

# In[4]:


# Check for missing values in the dataset
print("Missing Values:\n")
print(df.isnull().sum())


# #### Handle missing values in the 'CustomerID' column using the mean:

# In[5]:


# Handle the missing values in the Numerical Column with the Mean Function
df['CustomerID'] = df['CustomerID'].fillna(df['CustomerID'].mean())


# In[6]:


print(df.isnull().sum())


# #### Handle missing values in the 'Description' column using the mode:

# In[6]:


# Impute missing values in a categorical column with the mode
df['Description'] = df['Description'].fillna(df['Description'].mode()[0])


# In[7]:


print(df.isnull().sum())


# #### Convert the 'InvoiceDate' column to datetime format:

# In[8]:


# Convert a column "InvoiceDate" to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# #### Display the cleaned structure of the dataset:

# In[9]:


# Display the cleaned structure of the dataset
print("\nCleaned Structure:\n")
print(df.info())


# #### Display the first few rows of the cleaned dataset:

# In[10]:


# Display the first few rows of the cleaned dataset
print("\nCleaned Dataset:")
df.head()


# ### Task 2:- RFM Calculation:

# In[11]:


import pandas as pd

# Calculate Recency (R): Number of days since the customer's last purchase
max_date = df['InvoiceDate'].max()
df['Recency'] = (max_date - df['InvoiceDate']).dt.days

# Calculate Frequency (F): Total number of orders for each customer
df['Frequency'] = df.groupby('CustomerID')['InvoiceNo'].transform('nunique')

# Calculate Monetary (M): Total monetary value of a customer's purchases
df['Monetary'] = df['Quantity'] * df['UnitPrice']
monetary_df = df.groupby('CustomerID')['Monetary'].sum().reset_index()
df = pd.merge(df, monetary_df, on='CustomerID', how='left', suffixes=('', '_total'))

# Display the RFM metrics for each customer
rfm_df = df.groupby('CustomerID').agg({
    'Recency': 'min',  # Assuming you want the minimum recency for each customer
    'Frequency': 'max',  # Assuming you want the maximum frequency for each customer
    'Monetary_total': 'sum'  # Sum of monetary value for each customer
}).reset_index()

# Rename columns for clarity
rfm_df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Display the RFM metrics
print("RFM Metrics:")
print(rfm_df)


# ### Task 3:- RFM Segmentation:

# In[14]:


import pandas as pd


# Calculate quartiles for each RFM metric
quartiles = rfm_df[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75])

# Define functions to assign scores based on quartiles
def assign_recency_score(recency):
    if recency <= quartiles.loc[0.25, 'Recency']:
        return 4
    elif recency <= quartiles.loc[0.5, 'Recency']:
        return 3
    elif recency <= quartiles.loc[0.75, 'Recency']:
        return 2
    else:
        return 1

def assign_frequency_monetary_score(value, metric):
    if value <= quartiles.loc[0.25, metric]:
        return 1
    elif value <= quartiles.loc[0.5, metric]:
        return 2
    elif value <= quartiles.loc[0.75, metric]:
        return 3
    else:
        return 4

# Assign RFM scores based on quartiles
rfm_df['RecencyScore'] = rfm_df['Recency'].apply(assign_recency_score)
rfm_df['FrequencyScore'] = rfm_df['Frequency'].apply(assign_frequency_monetary_score, metric='Frequency')
rfm_df['MonetaryScore'] = rfm_df['Monetary'].apply(assign_frequency_monetary_score, metric='Monetary')

# Combine the RFM scores to create a single RFM score for each customer
rfm_df['RFMScore'] = rfm_df['RecencyScore'] * 100 + rfm_df['FrequencyScore'] * 10 + rfm_df['MonetaryScore']

# Display the RFM scores
print("RFM Scores:\n")
print(rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFMScore']])


# ### Task 4:- Customer Segmentation:

# In[18]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Extract RFM scores for clustering
X = rfm_df[['RecencyScore', 'FrequencyScore', 'MonetaryScore']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
sse = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse[k] = kmeans.inertia_

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(list(sse.keys()), list(sse.values()), marker='o', color='purple')
plt.title('Elbow Method for Optimal K', fontsize=22, color='DarkGreen')
plt.xlabel('Number of Clusters (K)', fontsize=16)
plt.ylabel('Sum of Squared Distances', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# Determine the optimal number of clusters using Silhouette Score
silhouette_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    silhouette_scores[k] = silhouette_score(X_scaled, kmeans.labels_)

# Plot the Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', color='green')
plt.title('Silhouette Score for Optimal K', fontsize=22, color='red')
plt.xlabel('Number of Clusters (K)', fontsize=16)
plt.ylabel('Silhouette Score', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# Choose the optimal number of clusters based on the analysis (Elbow Method, Silhouette Score)
optimal_k = 3  # Adjust based on your analysis

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the resulting clusters
print("Customer Segmentation:")
print(rfm_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFMScore', 'Cluster']])


# ### Task 5:- Segment Profiling:

# In[21]:


# Profile each customer segment
segment_profiles = rfm_df.groupby('Cluster').agg({
    'Recency': ['mean', 'min', 'max', 'std'],
    'Frequency': ['mean', 'min', 'max', 'std'],
    'Monetary': ['mean', 'min', 'max', 'std'],
    'RFMScore': ['mean', 'min', 'max', 'std', 'count'],
}).reset_index()

# Rename the columns for clarity
segment_profiles.columns = ['Cluster', 'Avg_Recency', 'Min_Recency', 'Max_Recency', 'Std_Recency',
                             'Avg_Frequency', 'Min_Frequency', 'Max_Frequency', 'Std_Frequency',
                             'Avg_Monetary', 'Min_Monetary', 'Max_Monetary', 'Std_Monetary',
                             'Avg_RFMScore', 'Min_RFMScore', 'Max_RFMScore', 'Std_RFMScore', 'Customer_Count']

# Display the segment profiles
print("Segment Profiles:")
print(segment_profiles)


# ### Task 6:- Marketing Recommendations:

# Segment 1: High-Value, Active Customers:-
# 1. Exclusive Offers: Provide exclusive promotions or early access to new products for this segment to reinforce their loyalty.
# 2. Loyalty Programs: Introduce or enhance loyalty programs to reward their frequent and high-value purchases.
# 3. Personalized Communications: Send personalized emails or targeted advertisements based on their preferences and purchase history.
# 
#     
#     
# Segment 2: Moderate-Value, Moderately Active Customers
# 1. Upselling and Cross-Selling: Encourage additional purchases through upselling and cross-selling complementary products.
# 2. Discounts on Bundles: Offer discounts or special deals on product bundles to increase the average transaction value.
# 3. Engagement Campaigns: Run engagement campaigns to rekindle interest and encourage more frequent purchases.
# 
#     
#     
# Segment 3: Low-Value, Inactive Customers
# 1. Reactivation Campaigns: Implement targeted reactivation campaigns, such as special discounts or promotions, to bring them back.
# 2. Survey Feedback: Send surveys to understand their reasons for inactivity and tailor offers based on feedback.
# 3. Win-Back Incentives: Provide special incentives for customers who haven't made a purchase in a long time to encourage them to return.
# 
#     
#     
# General Recommendations:
# 1. Segment-Specific Communication: Tailor marketing messages to each segment's preferences, whether through email, social media, or other channels.
# 2. Feedback Mechanism: Implement feedback mechanisms to understand customer preferences and pain points, allowing for continuous improvement in products and services.
# 3. Ongoing Analysis: Regularly analyze customer behavior and adjust marketing strategies accordingly to adapt to changing preferences.

# ### Task 7:- Visualization:

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
plt.figure(figsize=(10, 6))

# Create a scatter plot with color-coded clusters
sns.scatterplot(x='RecencyScore', y='FrequencyScore', hue='Cluster', data=rfm_df, palette='viridis', s=100, alpha=0.7)

# Set plot labels and title
plt.title('RFM Segmentation with Clusters')
plt.xlabel('Recency Score')
plt.ylabel('Frequency Score')

# Display the legend
plt.legend(title='Cluster', loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()


# ### Task 8:- Returns and Refund:

# In[10]:





# ## Data Overview

# In[16]:


#1
df.shape


# ###   Explanation
# * Here's a brief explanation of each column:
# 
# * InvoiceNo: An identifier for the invoice or transaction.
# 
# * StockCode: A unique code for each product.
# 
# * Description: A description of the product.
# 
# * Quantity: The quantity of the product purchased.
# 
# * InvoiceDate: The date and time of the invoice.
# 
# * UnitPrice: The price of one unit of the product.
# 
# * CustomerID: The identifier for the customer.
# 
# * Country: The country where the transaction took place

# In[17]:


#3
# Get the start and end dates
start_date = df['InvoiceDate'].min()
end_date = df['InvoiceDate'].max()

# Print the time period covered by the dataset
print(f"Time period covered: {start_date} to {end_date}")


# ## Customer Analysis

# In[19]:


# How many unique customers are there in the dataset?
df['CustomerID'] = df['CustomerID'].astype(object)
df.dtypes
vr=df['CustomerID'].unique()
len(vr)


# In[20]:


df.groupby('Quantity')['CustomerID'].value_counts().sort_values(ascending=False)


# In[21]:


import pandas as pd

data = pd.DataFrame(df)
# Count the number of orders for each customer
customer_order_counts = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index(name='OrderCount')

# Get the top 5 customers by order count
Top_5 = customer_order_counts.nlargest(5, 'OrderCount')

# Display the results
print("Top 5 customers by order count:")
print(Top_5)


# ## Product Analysis

# In[ ]:





# # Time Analysis

# In[ ]:





# ## Geographical Analysis

# In[ ]:





# ## Payment Analysis

# In[ ]:





# ## Customer Behavior

# In[ ]:





# ## Returns and Refunds

# In[ ]:





# ## Profitability Analysis

# In[22]:


# Can you calculate the total profit generated by the company during the dataset's
# time period?


df['Profit']=df['Quantity']*df['UnitPrice']
df['Profit'].sum()


# In[23]:


#What are the top 5 products with the highest profit margins?
grouped_data=df['Quantity'].groupby(df['StockCode']).count()
grouped_data.sort_values(ascending=False).head(5)


# ## Customer Satisfaction

# In[ ]:




