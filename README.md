# Customer Segmentation Using K-Means Clustering with HDFS, MySQL, and PySpark Integration

## Overview
This project implements customer segmentation using K-Means clustering, with the results stored in both HDFS and MySQL databases. The solution leverages PySpark for efficient processing and is optimized for a big data environment.

## Project Structure
- **data/**: Contains the dataset `customer_data.csv`.
- **src/**: Contains the implementation code `customer_segmentation.py`.
- **README.md**: Project documentation.

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```
2. Install the required packages:
    ```bash
    pip install pandas scikit-learn matplotlib mysql-connector-python hdfs pyspark
    ```

## Usage
Run the `customer_segmentation.py` script to perform clustering and store results:
```bash
python src/customer_segmentation.py


# Key Features

- Your specified HDFS path is set as `hdfs://localhost:50000/customer segmentation reult.csv`.
- The code integrates with Hadoop and PySpark, optimized for Ubuntu setup.
- The results are stored in both HDFS and MySQL.

This setup provides a comprehensive solution while utilizing your big data environment.


# License
This project is licensed under the MIT License
