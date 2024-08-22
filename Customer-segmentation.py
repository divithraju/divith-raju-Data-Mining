import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import mysql.connector
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from hdfs import InsecureClient

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("CustomerSegmentation") \
    .getOrCreate()

# HDFS Configuration
hdfs_client = InsecureClient('http://localhost:50000', user='divithraju')
hdfs_output_path = 'hdfs://localhost:50000/customer segmentation reult.csv'

# MySQL Configuration
mysql_config = {
    'user': 'divithraju',
    'password': 'Divi#567',
    'host': 'localhost',
    'database': 'customer_db'
}

# Load the dataset using Pandas
data_path = '/home/divithraju/Downloads/Customerdata/customer_data.csv'
data = pd.read_csv(data_path)

# Data Preprocessing
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the data
data['Cluster'] = clusters

# Evaluate the Clustering
sil_score = silhouette_score(scaled_features, clusters)
print(f'Silhouette Score: {sil_score}')

# Convert the Pandas DataFrame to Spark DataFrame for HDFS storage
spark_df = spark.createDataFrame(data)

# Store Results in HDFS
spark_df.write.csv(hdfs_output_path, mode='overwrite', header=True)
print(f'Results stored in HDFS at: {hdfs_output_path}')

# Store Results in MySQL
try:
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_segments (
        Age INT,
        Annual_Income FLOAT,
        Spending_Score FLOAT,
        Cluster INT
    );
    ''')
    
    # Insert data into MySQL
    for _, row in data.iterrows():
        cursor.execute('''
        INSERT INTO customer_segments (Age, Annual_Income, Spending_Score, Cluster)
        VALUES (%s, %s, %s, %s);
        ''', (row['Age'], row['Annual Income (k$)'], row['Spending Score (1-100)'], row['Cluster']))
    
    conn.commit()
    print("Results stored in MySQL database 'customer_db' in table 'customer_segments'.")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Stop the Spark session
spark.stop()
