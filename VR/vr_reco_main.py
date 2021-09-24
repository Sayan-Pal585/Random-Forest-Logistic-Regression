# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:33:20 2021

@author: SayanPal
"""

import scripts.recommendation.content_based as content_based
import os
import pandas as pd
import numpy as np
import pytd.pandas_td as td
import datetime as dt
print("Library imported")

database = os.environ['database']
output_table = os.environ['output_table']

lead_info, vin_info = content_based.import_tables()
lead, vin1_sparse = content_based.data_prep(lead_data=lead_info,vin_data=vin_info)

print("Unique PD VINs are {}".format(lead_info['pd_vin'].nunique()))
print("Unique VINS in Inventotry  are {}".format(vin_info['vin'].nunique()))
print("How many PD VINs are available in dealer inventory {}".format(len(set(lead_info['pd_vin'].unique().tolist()) & set(vin_info['vin'].unique().tolist()))))
print("{} % VINs in inventory dont have option_strings".format(round(vin_info['option_string'].isnull().sum()/len(vin_info)*100),2))
check  = lead_info[['pd_vin']].drop_duplicates().merge(vin_info[['vin','option_string']], left_on='pd_vin', right_on = 'vin', how ='inner')
print("{} % PD VINs dont have option_strings".format(round(check['option_string'].isnull().sum()/len(check)*100),2))
print("Data Preparation DONE")


print("Model Building Starts")
def run_app1():
  engine = td.create_engine('presto:{}'.format(database))
  con = td.connect()
  output = pd.DataFrame()
  for i in lead['pd_vin']:
    table = content_based.run_recom(voi = i, vin_ads = vin1_sparse)
    output=pd.concat([table,output])
  
  final_output = content_based.final_result(data = output, lead_value= lead_info, vin_value= vin_info, l =lead)
  print("Recommedation is available for {} explict VINs".format(len(final_output[final_output['recommendation1']!='N/A'])))
  td.to_td(final_output, '{}.{}'.format(database, 
  output_table), con, if_exists='append',index=False)
