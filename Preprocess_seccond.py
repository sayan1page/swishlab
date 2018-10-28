# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from textblob.classifiers import NaiveBayesClassifier

#classify the line in classifier c1 's target class 
def classify(line, c1):
	prob_dist = c1.prob_classify(line)
	return prob_dist.prob(1)


# Described in Preprocess first file 
def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].replace('{','').replace('}','').replace("'","").split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)


#Reading the input 
df = pd.read_csv('first.csv')

#transforming headline of news to it's probability of being "stock up"
#"stock up" is defined in updown column of data frame created at Preprocess first
#we are calculating probability using NaiveBayesClassifier
train = list(zip(df.headline, df.upordown))

c1 = NaiveBayesClassifier(train)

df['headline'] = classify(df['headline'],c1)

#splitting the multiple subject tags into multiple rows
new_rows = []
df.apply(splitListToRows,axis=1,args = (new_rows,"subjects",","))
new_df = pd.DataFrame(new_rows)


#splitting the multiple audience tags into multiple audiences
new_rows = []
new_df.apply(splitListToRows,axis=1,args = (new_rows,"audiences",","))
new_df1 = pd.DataFrame(new_rows)


# converting categorical column to numerical score
#We convert categorical feature values to numerical score by the average floor value of successful impressions where that feature value presents. 
#For example, numerical score for country =US is the average upordown column value  where Asset Code = US.
for col in new_df1.columns:
	if str(col) == "subjects" or str(col) == "audiences" or str(col) == "assetCodes":
		avgs = new_df1.groupby(col, as_index=False)['upordown'].aggregate(np.mean)
		for index,row in avgs.iterrows():
			k = row[col]
			v = row['upordown']
			new_df1.loc[new_df1[col] == k, col] = v


#save the transformed data
new_df1.to_csv('seccond.csv')





