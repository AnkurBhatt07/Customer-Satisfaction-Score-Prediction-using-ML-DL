Steps in data_analysis.ipynb

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


7. Note we have already done extensive data analysis and preprocessing experimentation in the other branch 'Initial_try' , so here we will straigtforward do the preprocessing based on that analysis.

8. created a missing indicator column for Customer City

9. Calculated city_counts : a dictionary containing each unique city as key and the number of records associated with the city in the entire data as value.
Here nan is a key which has 68000+ records associated with it.

10. saved the city_counts dictionary for later use.

11. Checked the different percentiles in the city_counts values data. 
    * 25% of the unique cities have only one record associated with them.
    * Median is 2 : meaning 50 % of the total unique cities have only 2 or less than 2 records associated with them.We can also infer like : 50% cities have only minimal complaints/issues with the eccomerce products.
    * MEAN is around 9.5. It shows that some cities have a higher complaints/issues placements.

12. Defined a function(city_bin) that provides a Numeric Category Bin for each city based on the number of records associated with the city.

13. mapped each city using a bin using the city_bin function to create a dictionary.

14. Create CityBins column using mapping.

15. Saved the mapping for cities with bin categories.

16. Scaled CityBins using standardScaler.

16. Saved the CityBinsScaler.

17. Calculated the mean csat for each city bin.Also calculated median csat for each city bin.
Here we see that for each city bin ,median csat score is 5.0.It shows 50% records of each category are associated with 5.0 csat score. 

18. Implemented Krushal test to see if the Csat score for each citybin are significantly different from other citybin groups or not. 
Here we find that there is significant difference in CSAT across CityBins.

19. Checked the distribution of connected_handling_time column.It shows right skewness.Range of the feature is from 0 to 2000 but most of the data lies in range 0 - 1100.

20. created a missing indicator column connectedTime_MI.

--------create bins thoughtfully
21. Defined a function to create time bins for connected handling time feature.

22. Created a new feature ConnectedTimeBins using the above function : connectedTimebins() as created time bins.

23. Scaled the ConnectedTimeBins to create connectedTimeBins_normalized.

24. Saved the connectedTimebins_scaler for later use.

25. Checked the normality of target for each time bin. No time bin was normally distributed.
Checked for significant difference between the CSAT scores for each time bins. Found no significant difference across citybins.

26. Checked the valuecounts of product category column.
Nan category has high number of records associated with it : 68711 records.

27. Visually inspected the distribution of csat scores for records with Nan as product category.
Peak is for 5.0 csat score followed by 4 and 1 respectively.
This distribution is almost similar to the csat scores distribution for records with non nan values as product category.

28. Created a Missing Indicator column for product category.

29. Product category has 9 unique categories.

30. Checked the product category and target relationship strength using Chi2 contingency test.
Found significant association with product category and csat score.

31. Replaced the missing values in product category with 'Unknown'.
Now total unique categories = 10.

32. One hot encoded the product category feature and saved the encoder as artifact.

33. Concatenated the one hot encoded product categories with data matrix.

34. 80 percent missing values in ItemPrice column.

35. Visualized the distribution of the Item Price column.
Highly right skewed.Peak around 0.

36. Created a missing indicator column for Item price named Price_MI.

37. The mean of the ItemPrice excluding the nan records is 5660.

38. Saved the lowest and highest values of Item price from the column as price_limits.joblib.


39. Imputed the missing prices records using the available prices data and the probability of occuring of each price also based on the data.


40. Checked the histogram plot again for imputed item price to see that shape of the distribution is much conserved even after imputations.

41. Transformed and Normalized the imputed Itemprice column using Power transformer.

42. Saved the transformer as joblib.

43. Checked the transformed and normalized Itemprice histogram . It is not right skewed and more normally distributed.

44. Concatenated normalized ItemPrice with the data matrix.

45. Dropped order Id and unique id columns.

46. Created a missing remarks indicator.

47. Get the Remarks sentiment polarity using text blob library.add the sentiment polarity as a new feature : Remarks_sentiment
----sentiment polarity is not scaled


48. Create Remark_word_count as a new feature.

49. Scales remark_word_count feature.And saved the scaler as joblib

50. Drop Survey_response_Date feature from data.

51. Drop Customer Remarks , Customer_City , connected_handling_time.

52. Checked the proportions of Channel Name categories using bar chart.

53. Checked the relationship between channel_name and csat score column using Chi square test : Found significant relationship between them.

Also implemented ANOVA : Implies mean of the csat score for atleast one of the channels is significantly different from other channel.

54. One hot encoded channel name.Saved the encoder as joblib.

55. Concatenated the channel_name encoded variable to the data_matrix.

56. Dropped the channel_name column from the data.

57. Checked the value_counts of the category feature.
50% of the records have Returns category.27% records are associated with order related category.Some category have very miniscule percentage of records associated (less than 1% records associated) namely Offers&Cashback , Others , App/website , Onboarding related

---------- We could have checked the missing prices percentage for each category.
58. Created a list containing names of those categories whose number of records are less than 10%.
Also created a list containing names of those categories whose number of records are less than 1%.

59. Checked the distribution of csat score across each category.


60. Using the Chi 2 test , found the significant relation between Category and CSAT Score.

61. One hot encoded Category feature.Saved the encoder.Concatenated the encoded variable to the data_matrix.

62. Subcategory column has high cardinality.Total 56 unique features.

Checked the relationship of Sub Category column with CSAT Score 

We see that 
    * 1/4 of all the Csat scores are associated with Reverse Pickup sub category.
    * Return request associates with 11% records associated with CSAT score = 5.
    * Delayed subcategory associated to 11% records with CSAT Score = 1.


63. Chi2 test shows significant association between sub category and csat score.

64. One hot encoded the subcategories column.Saved the encoder.

65. Concatenated the encoded variable with data_matrix.

66. Dropped the original category and sub category columns.

67. Dropped the original product category column.

68. order_date_time column has 68000+ null values.80% values missing.

69. Created a missing indicator column for order_date_time.
70. Created a new feature time_to_issue = data['Issue_reported at'] - data['order_date_time'] then converted it to hours.

71. Filled the missing time_to_issue with non missing time_to_issue using random.choice

72. Saved the min and max time_to_issue values in a dictionary and saved in joblib format.

73. Dropped the original order_date_time column.

74. Concatenate time_to_issue_hours to data_matrix.

75. Created Issue response time column using : data['IssueResponseTime'] = data['issue_responded'] - data['Issue_reported at'] then converted to hours.

76. Negative response time were replaced with random values taken from a normal distribution with mean = mean of non negative response times and std deviation = std dev of non negative response times.

77. Saved the Mean and std deviation of non negative response times as response time fallback.

78. Scaled the IssueresponsetimeHours feature.Then concatenated with the data_matrix.

79. Binned the agent name column by calculating the median csat score for each agent and creating a new column AgentRating representing these ratings for each agent.

80. Normalized the agent ratings using standardScaler.

81. Saved the Agent rating scaler object and the agent rating median csat mapper.

82. Added agent bins to data_matrix.

83. Created data_matrix1 which is copy of data_matrix.One hot encoded the supervisor feature and concatenated with the data_matrix1.Saved the encoder.

84. Created data_matrix2 which is copy of data_matrix.Created supervisor_rating by calculating mean csat for each supervisor and then scaling the ratings using standardScaler.
Concatenating with data_matrix2.

85. Saved the artifacts : mean csat rating for each supervisor and the scaler.

86. One hot encoded the manager feature adn concatenated with data_matrix1 and data_matrix2.

87. Ordinal encoded the tenure bucket feature and scaled it using standard scaler and concatenated with data_matrix1 and data_matrix2.
Saved the ordinal encoder and scaler.

88. One hot encoded the agent shift column and concatenated it with data_matrix1 and data_matrix2.saved the encoder.

89. Saved the data_matrix1 , data_matrix2 and y = numpy array of target feature(CSAT SCore)


90. Found the highly correlated feature using correlation matrix.Also using VIF to get data_matrix_reduced and data_matrix_reduced1 respectively.

91. saved the column indices removed using  both dimensionlity reduction approaches respectively.




















