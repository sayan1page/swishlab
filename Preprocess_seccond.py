# -*- coding: utf-8 -*-

import pandas as pd

# Described in Preprocess first file 
def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].replace('{','').replace('}','').replace("'","").split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
			
			
#Reading the input 
df = pd.read_csv('first.csv')


#splitting the multiple subject tags into multiple rows
new_rows = []
df.apply(splitListToRows,axis=1,args = (new_rows,"subjects",","))
new_df = pd.DataFrame(new_rows)


#splitting the multiple audience tags into multiple audiences
new_rows = []
new_df.apply(splitListToRows,axis=1,args = (new_rows,"audiences",","))
new_df1 = pd.DataFrame(new_rows)


#save the transformed data
new_df1.to_csv('seccond.csv')









