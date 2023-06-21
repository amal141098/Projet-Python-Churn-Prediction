#*Telecom Churn Prediction*



![img](/image/exit.png)

<hr>
### Table of content

<ul>
<span style="color:#0969da;">
<li><b> Explore the data</b></li>
<li><b> Machine Learning Prediction Analysis</b></li>
<li><b>Conclusions and recomendations</b></li>
</ul>
<hr>
##*Overview*

######This dataset comes from an Iranian telecom company, with each row representing a customer over a year period. 
######Along with a churn label, there is information on the customers' activity, such as call failures and subscription length.
######Not sure where to begin? Scroll to the bottom to find challenges!
<hr>
###Dataset:file_folder:

|    |   Call Failure |   Complaints |   Subscription Length |   Charge Amount |   Seconds of Use |   Frequency of use |   Frequency of SMS |   Distinct Called Numbers |   Age Group |   Tariff Plan |   Status |   Age |   Customer Value |   Churn |
|---:|---------------:|-------------:|----------------------:|----------------:|-----------------:|-------------------:|-------------------:|--------------------------:|------------:|--------------:|---------:|------:|-----------------:|--------:|
|  0 |              8 |            0 |                    38 |               0 |             4370 |                 71 |                  5 |                        17 |           3 |             1 |        1 |    30 |          197.64  |       0 |
|  1 |              0 |            0 |                    39 |               0 |              318 |                  5 |                  7 |                         4 |           2 |             1 |        2 |    25 |           46.035 |       0 |


### Data Dictionary

| Column Name           | Explanation                                          |
|-----------------------|------------------------------------------------------|
| Call Failure          | Number of call failures                              |
| Complaints            | Binary (0: No complaint, 1: Complaint)                |
| Subscription Length   | Total months of subscription                         |
| Charge Amount         | Ordinal attribute (0: Lowest amount, 9: Highest amount) |
| Seconds of Use        | Total seconds of calls                               |
| Frequency of Use      | Total number of calls                                |
| Frequency of SMS      | Total number of text messages                        |
| Distinct Called Numbers | Total number of distinct phone calls                 |
| Age Group             | Ordinal attribute (1: Younger age, 5: Older age)      |
| Tariff Plan           | Binary (1: Pay as you go, 2: Contractual)            |
| Status                | Binary (1: Active, 2: Non-active)                     |
| Age                   | Age of customer                                      |
| Customer Value        | The calculated value of customer                     |
| Churn                 | Class label (1: Churn, 0: Non-churn)                  |


##*Motivation

![img](/image/exit2.jpg) 

<ul>
<span>
<li>Im going to propose a model that predicts Customer Churn Better than a Baseline Classifier</li>
<li>Im going to undeline the most important causes related to customer Churn</b></li>
<li>In chapter 3 I will explore the data, showing appropriated graphs and making observations about the findings</b></li>
<li>In chapter 4 I will design the model and show its performance and the most important variables in churn customers</b></li>
<li>Finally in chapter 5 i will make conclusions and recomendations</b></li>
</ul>

<hr>

##Exploratory Data Analysis

#### Basic Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

####Datafrane definition

df=pd.read_csv("data/customer_churn.csv")

####Statistical description

|       |   Call Failure |   Complaints |   Subscription Length |   Charge Amount |   Distinct Called Numbers |   Age Group |   Tariff Plan |      Status |   Customer Value |       Churn |    SMS ratio |   Seconds_Use_Group |
|:------|---------------:|-------------:|----------------------:|----------------:|--------------------------:|------------:|--------------:|------------:|-----------------:|------------:|-------------:|--------------------:|
| count |     3150       | 3150         |            3150       |     3150        |                 3150      | 3150        |   3150        | 3150        |         3150     | 3150        | 3150         |          3150       |
| mean  |        7.62794 |    0.0765079 |              32.5419  |        0.942857 |                   23.5098 |    2.82603  |      1.07778  |    1.24825  |          470.973 |    0.157143 |    0.357711  |             1.49968 |
| std   |        7.26389 |    0.265851  |               8.57348 |        1.52107  |                   17.2173 |    0.892555 |      0.267864 |    0.432069 |          517.015 |    0.363993 |    0.297032  |             1.1185  |
| min   |        0       |    0         |               3       |        0        |                    0      |    1        |      1        |    1        |            0     |    0        |    0         |             0       |
| 25%   |        1       |    0         |              30       |        0        |                   10      |    2        |      1        |    1        |          113.801 |    0        |    0.0747664 |             0.25    |
| 50%   |        6       |    0         |              35       |        0        |                   21      |    3        |      1        |    1        |          228.48  |    0        |    0.319696  |             1       |
| 75%   |       12       |    0         |              38       |        1        |                   34      |    3        |      1        |    1        |          788.389 |    0        |    0.576657  |             2.75    |
| max   |       36       |    1         |              47       |       10        |                   97      |    5        |      2        |    2        |         2165.28  |    1        |    1         |             3       |


####Variables relationship and visual exploration

|   Age Group |   (Age, min) |   (Age, max) |
|------------:|-----------------:|-----------------:|
|           1 |               15 |               15 |
|           2 |               25 |               25 |
|           3 |               30 |               30 |
|           4 |               45 |               45 |
|           5 |               55 |               55 |

#### Analyze the average frequency of SMS and call usage among different age groups

|   Age Group |   Frequency of SMS |   Frequency of use |
|------------:|-------------------:|-------------------:|
|           1 |            20.1951 |            76.6423 |
|           2 |            75.4995 |            72.0752 |
|           3 |            90.0428 |            68.4428 |
|           4 |            42.0532 |            60.562  |
|           5 |            28.2471 |            77.5235 |


##Visualisation

######The distribution of Frequency of SMS
![img](/image/img1.png) 

######Frequency of SMS by age distribution
![img](/image/img2.png) 

#### Observation
Age Groups 4 and 5 uses comparatively much less SMS than Calls

###### Seconds of use by Age Group and Tariff Plan
![img](/image/img3.png) 
######Call Failure, Complains and Subscription Length vs Churn/No Churn
![img](/image/img4.png) 
![img](/image/img5.png) 
![img](/image/img6.png) 
#### Observation
All the customer that Churn had complains.

The customer mostly Churn between 32 and 38 months. The median is about 35 months.

The median of Call failures in customers that churns is about 5.

######Correlation between possible features and Features with Target

![img](/image/img7.png) 

#### Observation

Highly positively or negatively correlated features with Churn: Status, Complaints, Distinct Numbers, Freq of use, Customer Value

Highly positively correlated pairs of features: Age group with age, Freq of use with seconds of use, Customer value with freq of sms

##Machine Learning Prediction Analysis
######Number of Churn vs No Churn

![img](/image/img8.png) 

#### Observation

This is a highly imbalance problem. About 85% of targets labels are 0, what means that this percentage didn't churn. 

So a Zero rate Classifier that always predict a "No Churn" scenary would have an accuracy score close to 85%! So Our predictor have to do it better than that.

### :books: Machine learning model Definition 
#### Preprocessing the data
- Drop redundant columns
- Make new columns with the combination of others
- Convert seconds of use to categories according to bins
- Defining Target Variable and Features
- Instanciate Models
- Define Pipeline
- Test and Training Sets

###### Tuning the model

  ![img](/image/img9.png) 

######Feature Influence - What Causes or its related with Churn?

![img](/image/img10.png) 


##Final Conclusions
- The Model performed better than the baseline comparison model with an accuracy of about 95% versus 85%.
-  More important it correctly classified Customer Churn about 80% of the cases
- The most influencial variables related with Churn are: Complains - Status- Susbscription Lenght - Customer Value - Distinct Called Numbers - SMS ratio and Call Failure

##Actions that could prevent Churn

- Monitor Subcription Lenghts, because most of the Customers Churns appeared at about 32 to 38 months, and offering promotional discounts or benefits. The benefits could have been decided by age group taking on acount SMS vs Calls frequency (see Graphs in Exploratory Analysis)
- Tracking Call Failures if it is possible. When Call Failures exceed 5 contact the customer


<hr>
##Installation

The Code is written in Python 3.10.9 If you don't have Python installed you can find it [here](https://www.python.org/downloads/release/python-3109/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. 

