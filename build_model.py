# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.stats.stats import pearsonr

df = pd.read_csv("seccond.csv")

#preparing the input data to feed to model
y_train = df['upordown']
X_train = df[['assetCodes','headline', 'subjects', 'audiences', 'bodySize', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentPositive']]

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
				index.append(X.columns[i1-1])
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
		
		
		
#after this, one part is pending,exposed the model as a high performance rest api 
#which can response below 150ms latency all over the globe