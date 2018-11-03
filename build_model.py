# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr
import numpy as np
from textblob.classifiers import NaiveBayesClassifier
from sqlalchemy import create_engine
import sqlite3
import pickle

#create in memory db
db_name = "sqlite:///score.db"
disk_engine = create_engine(db_name)


#classify the line in classifier c1 's target class 
def classify(line, c1):
	prob_dist = c1.prob_classify(line)
	return prob_dist.prob(1)
	
df = pd.read_csv("seccond.csv")


#preparing the input data to feed to model
y_train = df['upordown']
X_back = df[['assetCodes','headline', 'subjects', 'audiences', 'bodySize', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentPositive']]

#transforming headline of news to it's probability of being "stock up"
#"stock up" is defined in updown column of data frame created at Preprocess first
#we are calculating probability using NaiveBayesClassifier
train = list(zip(df.headline, df.upordown))

c1 = NaiveBayesClassifier(train)

df['headline'] = classify(df['headline'],c1)

#save the classifier object
fileObject = open('classifier','wb') 
pickle.dump(c1 ,fileObject)  
fileObject.close()

# converting categorical column to numerical score
#We convert categorical feature values to numerical score by the average floor value of successful impressions where that feature value presents. 
#For example, numerical score for country =US is the average upordown column value  where Asset Code = US.
for col in df.columns:
	if str(col) == "subjects" or str(col) == "audiences" or str(col) == "assetCodes":
		avgs = df.groupby(col, as_index=False)['upordown'].aggregate(np.mean)
		for index,row in avgs.iterrows():
			k = row[col]
			v = row['upordown']
			df.loc[df[col] == k, col] = v

X = X_train = df[['assetCodes','headline', 'subjects', 'audiences', 'bodySize', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentPositive']]

X_train = X_train.astype(float) 
y_train = y_train.astype(float)
X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
		
index = []
i1 = 0
processed = 0

# drop the features which has low correlation with y_t
while(1):
	flag = True
	for i in range(X_train.shape[1]):
		if i > processed :
			#print(i1,X_train.shape[1],X.columns[i1])
			i1 = i1 + 1
			corr = pearsonr(X_train[:,i], y_train)
			PEr= .674 * (1- corr[0]*corr[0])/ (len(X_train[:,i])**(1/2.0))
			if corr[0] < PEr:
				X_train = np.delete(X_train,i,1)
				index.append(X_back.columns[i1-1])
				processed = i - 1 
				flag = False
				break
	if flag:
		break
	

#building simple linear regression model using tensorflow
#we are not using softmax regression model because that is not updatable
learning_rate = 0.0001
	
y_t = tf.placeholder("float", [None,1])
x_t = tf.placeholder("float", [None,X_train.shape[1]])
W = tf.Variable(tf.random_normal([X_train.shape[1],1],stddev=.01))
b = tf.constant(1.0)
	
model = tf.matmul(x_t, W) + b
cost_function = tf.reduce_sum(tf.pow((y_t - model),2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)
	
init = tf.initialize_all_variables()
	
with tf.Session() as sess:
	sess.run(init)
	w = W.eval(session = sess)
	of = b.eval(session = sess)
	print("Before Training #################################################")
	print(w,of)
	print("#################################################################")
	step = 0
	previous = 0
	while(1):
		step = step + 1
		sess.run(optimizer, feed_dict={x_t: X_train.reshape(X_train.shape[0],X_train.shape[1]), y_t: y_train.reshape(y_train.shape[0],1)})
		cost = sess.run(cost_function, feed_dict={x_t: X_train.reshape(X_train.shape[0],X_train.shape[1]), y_t: y_train.reshape(y_train.shape[0],1)})
		if step%1000 == 0:
			print(cost)
			if((previous- cost) < .0001):
				break
			previous = cost
	w = W.eval(session = sess)
	of = b.eval(session = sess)
	print("After Training #################################################")
	print(w,of)
	print("#################################################################")
	
	#store the weighted score to  in-memory db
	df2 = pd.DataFrame()
	started = False
	i = 0
	for col in X_back.columns:
		if (str(col) not in index) and (str(col) == "subjects" or str(col) == "audiences" or str(col) == "assetCodes"):
			#print(str(col),i)
			df1 = pd.DataFrame()
			df1['feature_value'] = X_back[col].apply(str).as_matrix()
			df1['feature_value'] = df1['feature_value'] + "_" + str(col)
			df1['score'] = np.multiply(X[col].astype(float),w[i][0])
			df1 = df1.drop_duplicates()
			if started:
				df2 = df2.append(df1)
			else:
				df2 = df1
			started = True
			i = i + 1
		print(df2)	
	df2.to_sql('scores', disk_engine, if_exists='replace')
			
		
		
