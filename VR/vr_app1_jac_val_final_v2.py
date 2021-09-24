# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:47:29 2020

@author: PritamDevadattaJena
This codes for all options codes and jaccard similarity range of 0.56 to less that 1

"""

#%%
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import jaccard_score
# Import the Python library for Treasure Data
import pytd.pandas_td as td
import os
from urllib import parse 
from getpass import getpass
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import multiprocessing as mp
from joblib import Parallel, delayed
print("All Libraries import done")

#%%
td_api_key = getpass('10729/001e998b60df46af8eef5e76045d22d66c7b5f17')

os.environ['TD_API_KEY'] = td_api_key
print("Succeeded to set the API key")

#os.environ["HTTP_PROXY"] = http_proxy = "http://cag\t8312pd:%s@iproxy.appl.chrysler.com:9090" % parse.quote_plus('Working@9826')
os.environ['TD_API_SERVER'] = endpoint = 'https://api.treasuredata.com'

con = td.connect(apikey=td_api_key, endpoint=endpoint)
                 #, http_proxy_proxy=http_proxy)

database = 'sandbox_data_science'
engine = td.create_engine('presto:{}'.format(database))
print("Connected to {}".format(database))

#%% Import Tables
query_lead_info = ("select * from sandbox_data_science.tmc_vr_mvp1_lead_info_testing")
lead_info = td.read_td(query_lead_info,engine)

query_vin_info = ("select * from sandbox_data_science.tmc_vr_mvp1_vin_info_testing")
vin_info = td.read_td(query_vin_info,engine)

#%%
lead = lead_info[['pd_vin']].drop_duplicates()
lead = lead.apply(lambda x: x.str.strip())

vin = vin_info[['vin','model','trim_description','franchise_id','make','year','option_string']]
vin = vin.fillna('NA')
vin = vin.apply(lambda x: x.str.strip())
#vin1_sparse = pd.concat([vin[['vin','model','trim_description','franchise_id','make']],vin['option_string'].str.get_dummies(sep=',').astype(np.int8)],axis=1)

#%% similarity

result = pd.DataFrame()

for item, i in enumerate(lead['pd_vin']):
    print(item)
    voi = i
    vin_dup = vin.copy()
    final={}
    final['voi']=[]
    final['rec_vin']=[]
    final['corr']=[]
    final['logic'] = []
    vin_dup = vin.copy()

    try:
        dealer = vin_dup[vin_dup['vin']==voi]['franchise_id'].reset_index(drop=True)[0]
        vin1_sparse = vin_dup[vin_dup['franchise_id'] == dealer].reset_index(drop =True)
        vin1_sparse = pd.concat([vin1_sparse[['vin','model','trim_description','franchise_id','make']],vin1_sparse['option_string'].str.get_dummies(sep=',').astype(np.int8)],axis=1)
        
        trim = vin1_sparse[vin1_sparse['vin']==voi]['trim_description'].reset_index(drop=True)[0]
        dealer = vin1_sparse[vin1_sparse['vin']==voi]['franchise_id'].reset_index(drop=True)[0]
        nameplate = vin1_sparse[vin1_sparse['vin']==voi]['model'].reset_index(drop=True)[0]
    
        data1 = vin1_sparse[vin1_sparse['vin']==voi].reset_index(drop=True)
        data1_sparse = data1.drop(data1.iloc[:,0:5],axis=1)
        data1_sparse = data1_sparse.loc[:, (data1_sparse == 1).all()]
        data1_sparse_list = data1_sparse.values.tolist()
    
        data2 = vin1_sparse[(vin1_sparse['trim_description']==trim) & (vin1_sparse['franchise_id']==dealer)]
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
            data3 = vin1_sparse[(vin1_sparse['model']==nameplate) & (vin1_sparse['franchise_id']==dealer)]
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
        print(i)
        pass



result = result.merge(lead_info[['pd_vin','offer_current','pc_rank']].drop_duplicates(),left_on = 'voi',right_on ='pd_vin', how='left') 
result = result.drop('pd_vin',axis=1)
result = result.merge(vin_info[['vin','internet_price']].drop_duplicates(),left_on = 'rec_vin',right_on ='vin', how='left') 
result = result.drop('vin',axis=1)
result['internet_price'] = result['internet_price'].astype('float64') 

# Score Constraint
result = result[(result['corr']>=0.55) & (result['corr']<1)] 
result = result.reset_index(drop=True)

# Price change constraint
result['price_change'] = np.abs(result['internet_price'] - result['offer_current'])/result['offer_current'] 
result = result[(result['price_change']<=0.1)] 
result = result.reset_index(drop=True)

# Top 2
result.sort_values(by = ['voi','corr','price_change'], ascending = [True, False, False], inplace =True)
result = result.groupby('voi').head(2).reset_index(drop=True)
  
# Preparing final table
recom = result[['voi','rec_vin']].groupby('voi')['rec_vin'].apply(lambda x: x.unique().tolist()).reset_index()
recom = recom.merge(result[['voi','corr']].groupby('voi')['corr'].apply(lambda x: x.tolist()).reset_index(), on ='voi', how='left')

recom = pd.concat([recom[['voi','corr']],pd.DataFrame(recom['rec_vin'].to_list(), columns=['recommendation1','recommendation2'])],axis= 1)    
recom = pd.concat([recom[['voi','recommendation1','recommendation2']],pd.DataFrame(recom['corr'].to_list(), columns=['jaccard1','jaccard2'])],axis= 1)    

recom = recom.merge(vin_info[['vin','franchise_id','dealer_location']].drop_duplicates(), left_on = 'voi', right_on ='vin',how ='left') 
#recom = recom.merge(lead_info[['pd_vin','pc_rank']].drop_duplicates(),left_on = 'voi',right_on ='pd_vin', how='left') 
recom = recom.drop(['vin'], axis = 1)
recom['dealer_location'] = recom['dealer_location'].astype('str')


#FIX
recom = result[['voi','rec_vin']].groupby('voi').aggregate(lambda x: x.unique().tolist()).reset_index()
recom = recom.merge(result[['voi','corr']].groupby('voi').aggregate(lambda x: x.tolist()).reset_index(), on ='voi', how='left')

result[['voi','corr']].groupby('voi').aggregate(lambda x: x.tolist()).reset_index()


#Exppp

result.rename(columns={'corr':'co_rr'},inplace = True)

recom = result[['voi','rec_vin']].groupby('voi')['rec_vin'].apply(lambda x: x.unique().tolist()).reset_index()
recom = recom.merge(result[['voi','co_rr']].groupby('voi')['corr'].apply(lambda x: x.tolist()).reset_index(), on ='voi', how='left')
