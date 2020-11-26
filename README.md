# RedPoints
==============================

This is a pilot project where data from customers suscribed to a service is analysed. The main goal is to to predict whether a given customer will churn or not, and we are also interested in extracting insights about why the customer might churn, not just a binary prediction. There are 3 original data files to be used all of them in the data folder. There are two other data files that can be generated from the original ones by running the notebook 'RedPoints.ipynb': 'contracts_full.csv' and 'group_by_contract.csv'. The former file is used as input into the notebook 'Classifier_bruteforce.ipynb' while the later into the notebook 'Classifier.ipynb'.

The 'RedPoint.ipynb' notebook addresses mainly the Data Exploration Analysis as well as Data Preprocessing for training the classifier, it focuses on the data file containing the main features and it generates, as mentioned, the two data files which are to be used for training the models.

The 'Classifier_bruteforce.ipynb' notebook uses the file 'contracts_full.csv' to train a very simple Logistic regression classifier, by using all features per date (i.e. 683 features = 34 dates * 20 time-dependent features + 3 time-independent features). 

The 'Classifier.ipynb' notebook trains a simple logistic regression as well but this time with the 'group_by_contract.csv' file (which contains only 103 features = 4 samples * 20 time-dependent features + 3 time-independent features). We use simple Logistic Regression to try to focus on the features and extracting insight from its relations.

Project Organization
------------

```
.
├── README.md
│ 
├── LICENSE
│ 
├── data
│   ├── CodingRound Processes.png
│   ├── test_conversion.csv
│   ├── test_dataset.csv
│   └── test_other_dataset.csv
│ 
├── contracts_full.csv
│ 
├── group_by_contract.csv
│ 
├── RedPoints.ipynb
│   
├── Classifier_bruteforce.ipynb
│   
├── Classifier.ipynb
│
└── aux_functions.py

```

```

```



