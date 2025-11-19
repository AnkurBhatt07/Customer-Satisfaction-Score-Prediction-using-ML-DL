Steps in data_analysis1.ipynb

1. Imported the necessary libraries for data wrangling , data visualization , path extraction , time formatting and artifacts saving.

2. Set up the paths to dataset , artifacts saving folders.

3. Make sure all the necessary paths and folders exist.If not create them.

4. Read the csv dataset.Check the data information and other aspects of the data

* The shape of the data is 85907 rows and 20 columns

* Date range of the data is 1st august to 31st august 2023.It is a one month period data.

* The Target variable is CSAT Score (1-5 scale)

* Columns with missing values are connected_handling_time , Customer_City , Product_category , Item_price , order_data_time , Customer Remarks and Order_id.

5. On checking the data distribution of the target variable(CSAT Score), we see that the data is very biased to some classes.
* Most of the records pertains to class 5 with a whopping 59000+ such records.
* Then the second majority of records is of class 1 and class 4 which 11000+ records each. 

* Class 3 and class 2 records are in minority with less that 3000 records for each of them.

* To get a better model that is robust and give good accuracy for each of the classes , it is necessary that we use class weights for penalty regularization.

Created class_weights based on the number of records of each csat score class.

