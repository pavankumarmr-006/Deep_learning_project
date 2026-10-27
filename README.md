Customer Support Ticket Classifier
1. Project Overview

This project prepares a dataset for classifying customer support tickets by category and priority. The dataset will later be used to build a machine learning model using TF-IDF and Logistic Regression.

2. Dataset Description
Source: Synthetically generated dataset created using Python for this learning project.
Total records: 600 tickets before any unexpected cleaning-related removals.
Columns:
text: Customer support message.
category: Type of customer issue.
priority: Urgency level of the issue.

The messages are artificial examples and do not represent real customer interactions.

3. Data Cleaning

The following preprocessing steps were performed:

Converted text, category, and priority labels to lowercase.
Removed leading and trailing whitespace.
Removed duplicate records.
Removed rows with missing required values.
4. Category Distribution
Category	Expected Count
Account	120
Billing	120
Delivery	120
Returns	120
Technical	120
Total	600
5. Priority Distribution
Priority	Expected Count
Low	200
Medium	200
High	200
Total	600
6. Dataset Split

The cleaned dataset was split using stratified sampling:

Training: 70% (420 tickets)
Validation: 15% (90 tickets)
Testing: 15% (90 tickets)

Stratification was performed using the combined category and priority labels to keep their combinations represented across the splits.

7. Generated Files
data/tickets.csv: Complete cleaned dataset.
data/train.csv: Training dataset.
data/validation.csv: Validation dataset.
data/test.csv: Test dataset.
8. Tools Used
Python
Pandas
Scikit-learn
9. Next Steps

The prepared dataset will be used for TF-IDF text vectorization and a Logistic Regression baseline model.