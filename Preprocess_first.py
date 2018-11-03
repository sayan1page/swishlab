# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

#If a data frame has a column containing multiple values of target_column separated by the separator, 
#then this method will create an array of rows of a data frame 
#where list in target_column transforms to multiple rows for each target_column value.
def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].replace('{','').replace('}','').replace("'","").split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
			

#Reading input data
df_news = pd.read_csv('news_sample.csv')

df_market = pd.read_csv('marketdata_sample.csv')

#stock is up if market open price is greater than maket closed price 

df_market['upordown'] = np.where(df_market['open'] < df_market['close'], '1', '0')

#split multiple asset codes to multiple rows
new_rows = []
df_news.apply(splitListToRows,axis=1,args = (new_rows,"assetCodes",","))
new_df_news = pd.DataFrame(new_rows)


df_market = df_market[['assetCode','upordown']]

#joining news data with market data by asset code and save it.

new_df_news = new_df_news[['assetCodes','headline', 'subjects', 'audiences', 'bodySize', 'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance', 'sentimentNegative', 'sentimentPositive']]

new_df_news['assetCodes'] = new_df_news['assetCodes'].astype(str)

df_market['assetCode'] = df_market['assetCode'].astype(str)

new_df_news['assetCodes'] = new_df_news['assetCodes'].str.strip()

df_market['assetCode'] = df_market['assetCode'].str.strip()

final_df_stage1 = pd.merge(df_market, new_df_news, left_on=['assetCode'], right_on=['assetCodes'], how='inner')

final_df_stage1.to_csv('first.csv')

#after running this code output table has only two rows both has 0 in upordown field. 
#We manually changes one value to 1 because we have to classify the data using upordown clolumn
#We know there is semi-supervised learning but due to time constrain we do not using that


