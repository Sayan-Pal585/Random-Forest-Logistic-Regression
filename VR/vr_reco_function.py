# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:32:50 2021

@author: SayanPal
"""

import sys
import os

#pip install necessary libraries
os.system(f"{sys.executable} -m pip install pytd --upgrade")

os.system(f"{sys.executable} -m pip install pandas==1.2.0")
os.system(f"{sys.executable} -m pip install numpy==1.19.5")

import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import jaccard_score
import pytd.pandas_td as td

#declare API Key and db/target tables
apikey = os.environ['TD_API_KEY'] 
database = os.environ['database']
input_table1 = os.environ['input_table1']
input_table2 = os.environ['input_table2']


def import_tables():
  engine = td.create_engine('presto:{}'.format(database))
  con = td.connect()
  
  query_lead_info = ("select * from {}").format(input_table1)
  lead_info = td.read_td(query_lead_info,engine)

  query_vin_info = ("select * from {}").format(input_table2)
  vin_info = td.read_td(query_vin_info,engine)
  return lead_info, vin_info
  

def data_prep(lead_data,vin_data):
  lead = lead_data[['pd_vin']].drop_duplicates()
  lead = lead.apply(lambda x: x.str.strip())
  
  vin = vin_data[['vin','model','trim_description','franchise_id','make','year','option_string']]
  vin = vin.fillna('NA')
  vin = vin.apply(lambda x: x.str.strip())
  
  return lead, vin
  
def run_recom(voi, vin_ads):
  result = pd.DataFrame()
  final={}
  final['voi']=[]
  final['rec_vin']=[]
  final['corr']=[]
  final['logic'] = []
  vin_dup = vin_ads

  try:
    
    dealer = vin_dup[vin_dup['vin']==voi]['franchise_id'].reset_index(drop=True)[0]
    sparse = vin_dup[vin_dup['franchise_id'] == dealer].reset_index(drop =True)
    sparse = pd.concat([sparse[['vin','model','trim_description','franchise_id','make']],sparse['option_string'].str.get_dummies(sep=',').astype(np.int8)],axis=1)
    
    trim = sparse[sparse['vin']==voi]['trim_description'].reset_index(drop=True)[0]
    nameplate = sparse[sparse['vin']==voi]['model'].reset_index(drop=True)[0]

    data1 = sparse[sparse['vin']==voi].reset_index(drop=True)
    data1_sparse = data1.drop(data1.iloc[:,0:5],axis=1)
    data1_sparse = data1_sparse.loc[:, (data1_sparse == 1).all()]
    data1_sparse_list = data1_sparse.values.tolist()  
    
    data2 = sparse[(sparse['trim_description']==trim) & (sparse['franchise_id']==dealer)]
    data2 = data2[~data2['vin'].isin([voi])].reset_index(drop=True)
    data2_sparse = data2.drop(data2.iloc[:,0:5],axis=1)
    data2_sparse = data2_sparse[data1_sparse.columns]
    data2_sparse_list = data2_sparse.values.tolist()
      
    if (len(data2))>=2:
      for j in range(len(data2)):
          corr = jaccard_score(data2_sparse_list[j],data1_sparse_list[0])
          final['voi'] = voi
          final['rec_vin'] = data2['vin'][j]
          final['corr'] = corr.squeeze()
          final['logic'] = 'trim'
          result = pd.concat([result,pd.DataFrame(final,index=[j])])
    else:
      data3 = sparse[(sparse['model']==nameplate) & (sparse['franchise_id']==dealer)]
      data3 = data3[~data3['vin'].isin([voi])].reset_index(drop=True)
      data3_sparse = data3.drop(data3.iloc[:,0:5],axis=1)
      data3_sparse = data3_sparse[data1_sparse.columns]
      data3_sparse_list = data3_sparse.values.tolist() 
      for k in range(len(data3)):
          corr = jaccard_score(data3_sparse_list[k],data1_sparse_list[0])
          final['voi'] = voi
          final['rec_vin'] = data3['vin'][k]
          final['corr'] = corr.squeeze()
          final['logic'] = 'nameplate'
          result = pd.concat([result,pd.DataFrame(final,index=[k])])

  except:
      print(voi)
      pass
  return result
  

def final_result(data, lead_value, vin_value,l):
  prepare_result = data.copy()  
  prepare_result = prepare_result.merge(lead_value[['pd_vin','offer_current']].drop_duplicates(),left_on = 'voi',right_on ='pd_vin', how='left') 
  prepare_result = prepare_result.drop('pd_vin',axis=1)
  prepare_result = prepare_result.merge(vin_value[['vin','internet_price']].drop_duplicates(),left_on = 'rec_vin',right_on ='vin', how='left') 
  prepare_result = prepare_result.drop('vin',axis=1)
  prepare_result['internet_price'] = prepare_result['internet_price'].astype('float64') 

# Score Constraint
  prepare_result = prepare_result[(prepare_result['corr']>=0.55) & (prepare_result['corr']<1)] 
  prepare_result = prepare_result.reset_index(drop=True)

# Price change constraint
  prepare_result['price_change'] = np.abs(prepare_result['internet_price'] - prepare_result['offer_current'])/prepare_result['offer_current'] 
  prepare_result = prepare_result[(prepare_result['price_change']<=0.1)] 
  prepare_result = prepare_result.reset_index(drop=True)

# Top 2 Recommendation
  prepare_result.sort_values(by = ['voi','corr','price_change'], ascending = [True, False, False], inplace =True)
  prepare_result = prepare_result.groupby('voi').head(2).reset_index(drop=True)
  prepare_result['corr'] = prepare_result['corr'].round(3)
  
# Preparing final table
  recom = prepare_result[['voi','rec_vin']].groupby('voi')['rec_vin'].apply(lambda x: x.unique().tolist()).reset_index()
  recom = recom.merge(prepare_result[['voi','corr']].groupby('voi')['corr'].apply(lambda x: x.tolist()).reset_index(), on ='voi', how='left')

  recom = pd.concat([recom[['voi','corr']],pd.DataFrame(recom['rec_vin'].to_list(), columns=['recommendation1','recommendation2'])],axis= 1)
  recom = pd.concat([recom[['voi','recommendation1','recommendation2']],pd.DataFrame(recom['corr'].to_list(), columns=['jaccard1','jaccard2'])],axis= 1)    
  
  recom = l[['pd_vin']].drop_duplicates().merge(recom, left_on='pd_vin', right_on ='voi', how='left')
  recom.drop(['voi'],axis=1,inplace=True)
  recom.rename(columns = {'pd_vin':'voi'},inplace=True)
  
  recom = recom.merge(vin_value[['vin','franchise_id','dealer_location']].drop_duplicates(), left_on = 'voi', right_on ='vin',how ='left') 
  recom = recom.drop(['vin'], axis = 1)
  
  recom['franchise_id'] = recom['franchise_id'].astype('str')
  recom['franchise_id'] = recom['franchise_id'] + str("$")  
  
  recom['dealer_location'] = recom['dealer_location'].astype('str')
  recom['dealer_location'] = recom['dealer_location'] + str("$")  
  
  recom = recom.replace(np.nan, 'N/A', regex=True)
  recom['run_date'] = dt.datetime.today().strftime("%Y-%m-%d")
  
  return recom
