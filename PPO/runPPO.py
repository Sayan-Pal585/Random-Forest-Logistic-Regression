# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:51:26 2020

Code to Run all 4 models [retail/Lease inmarket, retail/Lease Defection]
"""

import PPOModels
import PPOFunctions

#%% Run Final Models
retail_inmarket_output,retail_inmarket_decile, sign_retail_inmarket = PPOModels.RunRetailInmarket()    
lease_inmarket_output,lease_inmarket_decile, sign_lease_inmarket = PPOModels.RunLeaseInmarket()
retail_defection_output,retail_defection_decile, sign_retail_defection = PPOModels.RunRetailDefection()
lease_defection_output,lease_defection_decile,sign_lease_defection = PPOModels.RunLeaseDefection()

#%% Preparing outout tables
import pandas as pd
import numpy as np
retail = pd.merge(retail_defection_output,retail_inmarket_output,on = 'i_consmr', how = 'inner')
retail["flag"] = "is_retail"
lease = pd.merge(lease_inmarket_output,lease_defection_output,on = 'i_consmr',how = 'inner')
lease["flag"] = "is_lease"
con = pd.concat([retail, lease], ignore_index=True)
con['i_consmr'] = con['i_consmr'].astype(np.int64)

#%% Final Output Table 
va_data = PPOFunctions.VA_Results()
#ppo_score_data = PPOFunctions.PPO_Score()
ads = PPOFunctions.ppo_ads()
ads['i_consmr'] = ads['i_consmr'].astype(np.int64)

final_score = con.merge(va_data, on = 'i_consmr', how = 'left')

final_score = final_score[['i_consmr','flag','defection_prob_1', 'defection_decile',
                           'in_market_prob_1', 'in_market_decile','current_brand', 
                           'current_nameplate', 'current_trim', 'first_preferred_nameplate', 
                           'first_preferred_brand', 'first_preferred_score', 'second_preferred_nameplate', 
                           'second_preferred_brand', 'second_preferred_score', 'third_preferred_nameplate', 
                           'third_preferred_brand', 'third_preferred_score']]

final_score = final_score.rename(columns={'defection_prob_1':'defection_prob',
                                          'in_market_prob_1':'in_market_prob'})

final_score_v1 = final_score.merge(ads, on = 'i_consmr', how = 'inner')

final_score_tab = final_score_v1[['i_consmr','hh_id','flag','defection_prob', 'defection_decile',
                           'in_market_prob', 'in_market_decile','current_brand', 
                           'current_nameplate', 'current_trim', 'first_preferred_nameplate', 
                           'first_preferred_brand', 'first_preferred_score', 'second_preferred_nameplate', 
                           'second_preferred_brand', 'second_preferred_score', 'third_preferred_nameplate', 
                           'third_preferred_brand', 'third_preferred_score']]
#%% Writing Score table to sandbox_data_science
import pytd.pandas_td as td
con_new = td.connect()
td.to_td(con, '{}.{}'.format('sandbox_data_science','tmc_ppo_2020_score_v2'), con_new, if_exists='replace',index=False)
#%% Writing Final Score table to sandbox_data_science
import pytd.pandas_td as td
con_new = td.connect()
td.to_td(final_score_tab, '{}.{}'.format('sandbox_data_science','tmc_ppo_result_20200915'), con_new, if_exists='replace',index=False)
#%%