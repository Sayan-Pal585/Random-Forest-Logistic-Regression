
import SOCIModels
import SOCIFunctions
import pandas as pd
import pytd.pandas_td as td
import os
from urllib import parse

#%%
#--------------------------VISIT-----------------------------------------
bnso_r_output_visit,bnso_r_output_feature_visit = SOCIModels.bnso_retail()
bnso_l_output_lease,bnso_l_output_feature_visit = SOCIModels.bnso_lease()
buso_wo_output_visit,buso_wo_output_feature_visit = SOCIModels.buso_wo_service()
buso_ws_output_visit,buso_ws_output_feature_visit = SOCIModels.buso_ws_service()

# #-----------------------------Spend--------------------------------------------
bnso_output, bnso_output_feature = SOCIModels.RunBNSOSpend()
buso_ws_output, buso_ws_output_feature = SOCIModels.RunBUSO_ws_Spend()
buso_wo_output, buso_wo_output_feature = SOCIModels.RunBUSO_wo_Spend()

# ##Concat spends results 
final_spend = pd.concat([bnso_output, buso_ws_output, buso_wo_output],ignore_index=True, axis =0)

# # Concat Visit results
final_visit = pd.concat([bnso_r_output_visit,bnso_l_output_lease,buso_wo_output_visit,buso_ws_output_visit],ignore_index=True,axis=0)


# Preparing Final Table
final = final_spend.merge(final_visit,on=["fca_id","vin"])
final['q_spend'] = final.apply(SOCIFunctions.bucket_list, axis=1)
final["q_combo"] = final["q_spend"] * final["visit_prob_1"]
final.rename(columns={"visit_prob_1":"q_visit"},inplace=True)
final = final.drop(['spend_group','visit_prob_0'], axis=1)

# # Preparing final mailable table
# final_mailable = final[[]]

#%%
# Upload the final table
td_api_key = '10729/718b8e1f45ab24b17cabb39f75aabc37cf49bfbd'

os.environ['TD_API_KEY'] = td_api_key
print("\n Succeeded to set the API key")

os.environ["HTTP_PROXY"] = http_proxy = "http://cag\t8318sp:%s@iproxy.appl.chrysler.com:9090" % parse.quote_plus('Milu9007423695@')
os.environ['TD_API_SERVER'] = endpoint = 'https://api.treasuredata.com'

con = td.connect(apikey=td_api_key, endpoint=endpoint, http_proxy_proxy=http_proxy)

database = 'sandbox_data_science'
engine = td.create_engine('presto:{}'.format(database))
print("\n Connected to {}".format(database))
print("\n Connection Establish ")

#%%
con_new = td.connect()
td.to_td(final, '{}.{}'.format('sandbox_data_science','tmc_mopar_soci_oct_all_20201024'), con_new, if_exists='replace',index=False)
# td.to_td(final_mailable, '{}.{}'.format('sandbox_data_science','tmc_mopar_soci_oct'), con_new, if_exists='replace',index=False)
