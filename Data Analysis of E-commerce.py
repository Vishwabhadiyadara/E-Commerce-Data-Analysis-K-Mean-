#!/usr/bin/env python
# coding: utf-8

# # 1. Data Overview
# o What is the size of the dataset in terms of the number of rows and columns?
# o Can you provide a brief description of each column in the dataset?
# o What is the time period covered by this dataset?

# ### Data Overview
# Total Entries (Rows): 541,909
# Total Columns: 8
# Column Descriptions:
# InvoiceNo: The invoice number (object type, possibly alphanumeric).
# StockCode: Product item code (object type).
# Description: Description of the product (object type).
# Quantity: Quantity of each product per transaction (integer type).
# InvoiceDate: The date and time of the invoice (object type, formatted as a string).
# UnitPrice: Unit price of the product (float type).
# CustomerID: Identifier for the customer (float type, but more likely to represent an integer ID).
# Country: Country name where the customer is located (object type).
# Time Period Covered:
# The time period covered by the dataset is from December 1, 2010, at 08:26 to December 9, 2011, at 12:50.

# # 2. Customer Analysis
# o How many unique customers are there in the dataset?
# o What is the distribution of the number of orders per customer?
# o Can you identify the top 5 customers who have made the most purchases by order

# In[2]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 1. Count the number of unique customers
unique_customers = data['CustomerID'].nunique()

# 2. Distribution of the number of orders per customer
orders_per_customer = data.groupby('CustomerID')['InvoiceNo'].nunique()
orders_distribution = orders_per_customer.value_counts()

# 3. Top 5 customers who have made the most purchases by order count
top_5_customers_by_orders = orders_per_customer.sort_values(ascending=False).head(5)

# Formatting and outputting the results
print("Customer Analysis Results:")
print("-" * 30)
print(f"Number of Unique Customers: {unique_customers}\n")
print("Distribution of the Number of Orders per Customer:")
print(orders_distribution, "\n")
print("Top 5 Customers by Order Count:")
print(top_5_customers_by_orders)


# # 3. Product Analysis
# o What are the top 10 most frequently purchased products?
# o What is the average price of products in the dataset?
# o Can you find out which product category generates the highest revenue?

# In[5]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Adding a revenue column
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# 1. Top 10 Most Frequently Purchased Products
top_10_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# 2. Average Price of Products
average_price = data['UnitPrice'].mean()

# 3. Product Category Generating the Highest Revenue
# Using 'Description' as a proxy for product categories
highest_revenue_product = data.groupby('Description')['Revenue'].sum().sort_values(ascending=False).head(1)

# Formatting and outputting the results
print("Product Analysis Results:")
print("-" * 30)
print("Top 10 Most Frequently Purchased Products:")
print(top_10_products, "\n")
print(f"Average Price of Products: {average_price:.2f}\n")
print("Product Category Generating the Highest Revenue:")
print(highest_revenue_product)


# # 4. Time Analysis
# o Is there a specific day of the week or time of day when most orders are placed?
# o What is the average order processing time?
# o Are there any seasonal trends in the dataset?

# In[7]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Convert 'InvoiceDate' to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Calculating 'Revenue'
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# 1. Specific Day of the Week or Time of Day for Most Orders
data['DayOfWeek'] = data['InvoiceDate'].dt.day_name()
data['HourOfDay'] = data['InvoiceDate'].dt.hour
orders_by_day = data['DayOfWeek'].value_counts()
orders_by_hour = data['HourOfDay'].value_counts().sort_index()

# 2. Average Order Processing Time
# Note: This requires order completion time, which might not be available in your dataset.

# 3. Seasonal Trends
# Extracting month and year for seasonal trend analysis
data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')
seasonal_trends_quantities = data.groupby('YearMonth')['Quantity'].sum()
seasonal_trends_revenues = data.groupby('YearMonth')['Revenue'].sum()

# Formatting and outputting the results
print("Time Analysis Results:")
print("-" * 30)
print("Orders by Day of the Week:")
print(orders_by_day, "\n")
print("Orders by Hour of the Day:")
print(orders_by_hour, "\n")
print("Seasonal Trends (Order Quantities):")
print(seasonal_trends_quantities, "\n")
print("Seasonal Trends (Revenues):")
print(seasonal_trends_revenues)


# # 5. Geographical Analysis
# o Can you determine the top 5 countries with the highest number of orders?
# o Is there a correlation between the country of the customer and the average order
# value?

# In[8]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Calculating 'Revenue'
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# 1. Top 5 Countries with the Highest Number of Orders
top_5_countries_orders = data['Country'].value_counts().head(5)

# 2. Average Order Value for Each Country
# Calculating total revenue and number of orders for each country
country_revenues = data.groupby('Country')['Revenue'].sum()
country_order_counts = data.groupby('Country')['InvoiceNo'].nunique()

# Calculating average order value for each country
average_order_value_by_country = country_revenues / country_order_counts

# Sorting to see the countries with the highest average order values
sorted_average_order_values = average_order_value_by_country.sort_values(ascending=False)

# Formatting and outputting the results
print("Geographical Analysis Results:")
print("-" * 30)
print("Top 5 Countries by Number of Orders:")
print(top_5_countries_orders, "\n")
print("Average Order Value by Country:")
print(sorted_average_order_values)


# # 6. Payment Analysis
# o What are the most common payment methods used by customers?
# o Is there a relationship between the payment method and the order amount?

# In[ ]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Ensure you have a 'PaymentMethod' column in your data. If not, replace 'PaymentMethod' with the correct column name.
# Calculating 'Revenue'
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# 1. Most Common Payment Methods Used by Customers
common_payment_methods = data['PaymentMethod'].value_counts()

# 2. Relationship Between Payment Method and Order Amount
# Grouping data by payment method and calculating average revenue
average_revenue_by_payment_method = data.groupby('PaymentMethod')['Revenue'].mean()

# Formatting and outputting the results
print("Payment Analysis Results:")
print("-" * 30)
print("Most Common Payment Methods:")
print(common_payment_methods, "\n")
print("Average Revenue by Payment Method:")
print(average_revenue_by_payment_method)


# ans 6.The KeyError you encountered indicates that the 'PaymentMethod' column is not found in your dataset. As I mentioned earlier, the analysis of payment methods requires specific data about the payment methods used for each transaction, which seems to be missing in your dataset.
# 
# 

# # 7. Customer Behavior
# o How long, on average, do customers remain active (between their first and last
# purchase)?
# o Are there any customer segments based on their purchase behavior?

# In[10]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Convert 'InvoiceDate' to datetime format
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# 1. Average Duration of Customer Activity
# Finding the first and last purchase dates for each customer
first_purchase = data.groupby('CustomerID')['InvoiceDate'].min()
last_purchase = data.groupby('CustomerID')['InvoiceDate'].max()

# Calculating the duration of activity for each customer
customer_activity_duration = last_purchase - first_purchase

# Calculating the average duration of customer activity
average_activity_duration = customer_activity_duration.mean()

# 2. Customer Segments Based on Purchase Behavior
# Counting the number of purchases for each customer
purchase_counts = data.groupby('CustomerID')['InvoiceNo'].nunique()

# Segmenting customers based on the number of purchases (simple segmentation)
# For example, defining three segments: low, medium, high
low_activity = purchase_counts[purchase_counts <= purchase_counts.quantile(0.33)]
medium_activity = purchase_counts[(purchase_counts > purchase_counts.quantile(0.33)) & (purchase_counts <= purchase_counts.quantile(0.66))]
high_activity = purchase_counts[purchase_counts > purchase_counts.quantile(0.66)]

# Formatting and outputting the results
print("Customer Behavior Analysis Results:")
print("-" * 30)
print(f"Average Duration of Customer Activity: {average_activity_duration}")
print("\nCustomer Segments Based on Purchase Behavior:")
print("Low Activity Segment (Bottom 33%):", len(low_activity))
print("Medium Activity Segment (Middle 33%):", len(medium_activity))
print("High Activity Segment (Top 33%):", len(high_activity))


# # 8. Returns and Refunds
# o What is the percentage of orders that have experienced returns or refunds?
# o Is there a correlation between the product category and the likelihood of returns?

# In[3]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 1. Percentage of Orders with Returns or Refunds
# Assuming returns are indicated by a negative quantity
returns_data = data[data['Quantity'] < 0]
total_orders = len(data['InvoiceNo'].unique())
total_returns = len(returns_data['InvoiceNo'].unique())
percentage_returns = (total_returns / total_orders) * 100

# 2. Correlation Between Product Category and Returns
# Assuming 'Description' serves as a product category
# Calculating the number of returns for each product category
returns_by_category = returns_data['Description'].value_counts()

# Calculating the total number of orders for each product category
total_orders_by_category = data['Description'].value_counts()

# Calculating the return rate for each category
return_rate_by_category = (returns_by_category / total_orders_by_category) * 100

# Sorting categories by return rate
sorted_return_rates = return_rate_by_category.sort_values(ascending=False)

# Formatting and outputting the results
print("Returns and Refunds Analysis Results:")
print("-" * 30)
print(f"Percentage of Orders with Returns or Refunds: {percentage_returns:.2f}%\n")
print("Return Rate by Product Category:")
print(sorted_return_rates)


# # 9. Profitability Analysis
# o Can you calculate the total profit generated by the company during the dataset's
# time period?
# o What are the top 5 products with the highest profit margins?

# In[12]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Calculating 'Revenue'
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# 1. Total Profit Generated
# Assuming 'Revenue' as a proxy for profit, in absence of COGS data
total_profit = data['Revenue'].sum()

# 2. Top 5 Products with the Highest Profit Margins
# Assuming 'UnitPrice' represents the selling price, and ignoring the actual cost
# Profit margin calculation here is an approximation
# Calculating total revenue and total quantity for each product
total_revenue_per_product = data.groupby('Description')['Revenue'].sum()
total_quantity_per_product = data.groupby('Description')['Quantity'].sum()

# Calculating approximate profit margin per product (Revenue / Quantity)
approximate_profit_margin_per_product = total_revenue_per_product / total_quantity_per_product

# Identifying the top 5 products with the highest approximate profit margins
top_5_products_by_profit_margin = approximate_profit_margin_per_product.sort_values(ascending=False).head(5)

# Formatting and outputting the results
print("Profitability Analysis Results:")
print("-" * 30)
print(f"Total Profit Generated: {total_profit:.2f}\n")
print("Top 5 Products by Approximate Profit Margin:")
print(top_5_products_by_profit_margin)


# # 10. Customer Satisfaction
# o Is there any data available on customer feedback or ratings for products or services?
# o Can you analyze the sentiment or feedback trends, if available?

# In[13]:


import pandas as pd

# Load the dataset with an alternate encoding
file_path = 'data.csv'  # Replace with your file path
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Check for customer feedback or ratings data
feedback_columns = [col for col in data.columns if 'Review' in col or 'Rating' in col]
has_feedback_data = len(feedback_columns) > 0

# Analyze feedback or ratings if available
if has_feedback_data:
    print("Customer Feedback Analysis Results:")
    print("-" * 30)

    # If there's a 'Rating' column
    if 'Rating' in feedback_columns:
        average_rating = data['Rating'].mean()
        print(f"Average Product Rating: {average_rating:.2f}")

    # If there's a 'CustomerReview' column, basic sentiment analysis could be performed
    # Here, you would typically use a library like NLTK or TextBlob
    # For this example, we're not performing textual sentiment analysis due to its complexity
    # and the need for additional libraries

    # Add your code for textual sentiment analysis here if needed

else:
    print("No customer feedback or rating data available in the dataset.")

