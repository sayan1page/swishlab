# -*- coding: utf-8 -*-

import falcon
from falcon_cors import CORS
import json
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import math
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.engine import Engine
from sqlalchemy import event
import pickle
import math
#This is a high performance rest api to expose the model
#set the memory limit of in memory db
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA cache_size=100000")
    cursor.close()

class Predictor(object):
	def __init__(self):
		#loading in memory db
		db = create_engine('sqlite:///score.db')
		metadata = MetaData(db)
		self.scores = Table('scores', metadata, autoload=True)
		Session = sessionmaker(bind=db)
		self.session = Session()
		#loading headline classifier
		fileObject = open('classifier','rb') 
		self.cl = pickle.load(fileObject) 
		fileObject.close()
		

	# fetch the score for feature values from in memory db
	def get_score(self, session, scores,feature_values):
		s = session.query(func.sum(scores.c.score).label('sum')).filter(scores.c.feature_value.in_(feature_values)).one()
		if s is not None:
			if s[0] is not None:
				return s[0]
		return 0

	#process the fetaure values
	def process_value(self, f, value):
		feature_values = []
		for v in value:
			feature_values.append(str(v) + "_" + str(f))
		return feature_values	

	#tansform the score to probability
	def transform(self,p):
		return math.exp(p) / (1 + math.exp(p))

	def on_post(self, req, resp):
		try:
			# read input
			input_json = json.loads(str(req.stream.read().decode("utf-8")))
			feature_values = []	
			predicted = 0
		
			#process the categorical parameters in parallel
			pool = ThreadPoolExecutor(max_workers=3)
			for f in input_json:
				if f == "subjects" or f == "audiences" or f == "assetCodes":
					future = pool.submit(self.process_value,f,input_json[f])
					feature_values.extend(future.result())
				else:
					if f == 'headline':
						#process the headline
						predicted = predicted + self.cl.prob_classify(input_json[f]).prob(1)
					else:
						#process the numeric parameter
						predicted = predicted + input_json[f]
			pool.shutdown(wait = True)
			#fetch the scores for categorical features
			predicted = predicted + self.get_score(self.session, self.scores, feature_values)
		
			resp.status = falcon.HTTP_200
			res = self.transform(predicted)
			resp.body = str(res)
		except:
			resp.status = falcon.HTTP_500
			resp.body = "There is internal error"
    


cors = CORS(allow_all_origins=True,allow_all_methods=True,allow_all_headers=True)  
wsgi_app = api = falcon.API(middleware=[cors.middleware])
p = Predictor()

api.add_route('/predict', p)

#input {"audiences": ["BSW","CNR"], "subjects":["ENT","US"], "headline":"This is Sayan","sentimentPositive":0.2}
#url : http://localhost:8080/predict
#output 0.7683156853589814 