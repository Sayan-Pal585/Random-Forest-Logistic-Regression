# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:04:24 2020

THis code has all the functions related to loading data, pre-processing data, outlier detection and model building
"""

#%% Loading All packages
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import the Python library for Treasure Data
import pytd.pandas_td as td
import os
from urllib import parse 
from getpass import getpass
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import recall_score, precision_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import math

print("\n All Libraries import done")

#%% Building Conncetion

With Databases.

#%% Loading ALl tables

def retail_inmarket_data():
    retail_inmarket_train_script = ("""select retail_is_in_market,nemails_opened_last_1yr,nvehiclesowned_new_active_last_5yrs,
                      nservicesdone_forpurchase_new_active_last_5yrs,ndayssince_lastservice_forpurchase_new,
                      spend_onservice_forowned_new_last_5yr,nLeads_past90Days,totalspend_onservice_forowned_new_last_5yr,
                      avg_service_downtime_for_owned_new,ndayssince_last_purchase_new,ndayssince_lastpurchase_brand_jeep,
                      ndayssince_lastpurchase_brand_dodge,ndayssince_lastpurchase_brand_ram 
                      from sandbox_data_science.tmc_modelbase_ActiveRetail_out_of_time_validation_test_train_v2 
                      where is_active=1 and is_retail=1 and is_new=1""")
    retail_inmarket_train = td.read_td(retail_inmarket_train_script,engine)
    retail_inmarket_score_script = ("""select i_consmr,nemails_opened_last_1yr,nvehiclesowned_new_active_last_5yrs,
                      nservicesdone_forpurchase_new_active_last_5yrs,ndayssince_lastservice_forpurchase_new,
                      spend_onservice_forowned_new_last_5yr,nLeads_past90Days,totalspend_onservice_forowned_new_last_5yr,
                      avg_service_downtime_for_owned_new,ndayssince_last_purchase_new,ndayssince_lastpurchase_brand_jeep,
                      ndayssince_lastpurchase_brand_dodge,ndayssince_lastpurchase_brand_ram 
                      from sandbox_data_science.tmc_ppo_ADS_MVP2_20200914 
                      where is_active=1 and is_retail=1 and is_new=1""")
    retail_inmarket_score = td.read_td(retail_inmarket_score_script,engine)        
    return retail_inmarket_train, retail_inmarket_score


def lease_inmarket_data():
    lease_inmarket_train_script = ("""select lease_is_in_market,nemails_opened_last_1yr,ndayssince_lastservice_forlease_new,
                    nvehiclesleased_new_lt,spend_onservice_forleased_new_last_5yr,navgdaysbetween_services_forlease_new_lt,
                    nleads_past90days,totalspend_onservice_forleased_new_last_5yr,avg_service_downtime_for_leased_new,
                    lease_ending_in_next_six_months from tmc_modelbase_ActiveLeasers_out_of_time_validation_test_train
                    where is_active = 1 and is_lease = 1 and is_new = 1""")
    lease_inmarket_train = td.read_td(lease_inmarket_train_script,engine)
    lease_inmarket_score_script = ("""select i_consmr,nemails_opened_last_1yr,ndayssince_lastservice_forlease_new,nvehiclesleased_new_lt,
                    spend_onservice_forleased_new_last_5yr,navgdaysbetween_services_forlease_new_lt,
                    nleads_past90days,totalspend_onservice_forleased_new_last_5yr,avg_service_downtime_for_leased_new,
                    lease_ending_in_next_six_months 
                    from tmc_ppo_ADS_MVP2_20200914
                    where is_active = 1 and is_lease = 1 and is_new = 1""")
    lease_inmarket_score = td.read_td(lease_inmarket_score_script,engine)
    return lease_inmarket_train, lease_inmarket_score


def retail_defection_data():
    retail_defection_train_script = ("""select did_churn_retail,avg_service_downtime_purchased_active_std,change_in_avg_snowfall_retail_std,
                        days_between_recalls_purchased_active_std,days_since_first_recall_purchased_active_std,interest_present_in_hh_in_auto_work,
                        interest_present_in_hh_in_usa_travel,is_recall_type_california_emission_retail_std,
                        is_recall_type_customer_satisfaction_retail_std,is_recall_type_federal_emission_retail_std,
                        is_recall_type_safety_retail_std,is_retail_active_recall_closed_std,
                        is_retail_active_recall_open_sseo_std,is_retail_active_recall_open_std,
                        days_since_last_service_purchased_active_std,nemails_opened_last_1yr_std,
                        nLeads_past90Days,nRecalls_purchased_active_std,
                        avg_nservices_purchased_active_std,nvehicles_purchased_active_std,days_between_purchased_disposals_std,
                        days_since_last_purchased_disposal_std,navgdaysbetween_services_forretail_new_std ,
                        nvehicles_brand_dodge_purchased_active_std,nvehicles_brand_jeep_purchased_active_std,
                        nvehicles_brand_ram_purchased_active_std,nvehicles_purchased_std,
                        perc_total_spend_customer_paid_purchased_active_std,perc_total_spend_replacing_parts_purchased_active_std,
                        nVins_disposed_std,is_segment_c_std,mileage_recorded_from_last_service_for_active_owned_std
                        from tmc_ppo_ADS_MVP2_retail_std_20190930""")
    retail_defection_train = td.read_td(retail_defection_train_script,engine)
    retail_defection_score_script = ("""select i_consmr,avg_service_downtime_purchased_active_std,
                        change_in_avg_snowfall_retail_std,days_between_recalls_purchased_active_std,
                        days_since_first_recall_purchased_active_std,interest_present_in_hh_in_auto_work,
                        interest_present_in_hh_in_usa_travel,is_recall_type_california_emission_retail_std,
                        is_recall_type_customer_satisfaction_retail_std,is_recall_type_federal_emission_retail_std,
                        is_recall_type_safety_retail_std,is_retail_active_recall_closed_std,
                        is_retail_active_recall_open_sseo_std,is_retail_active_recall_open_std,
                        days_since_last_service_purchased_active_std,nemails_opened_last_1yr_std,
                        nLeads_past90Days,nRecalls_purchased_active_std,avg_nservices_purchased_active_std,
                        nvehicles_purchased_active_std,days_between_purchased_disposals_std,
                        days_since_last_purchased_disposal_std,navgdaysbetween_services_forretail_new_std ,
                        nvehicles_brand_dodge_purchased_active_std,nvehicles_brand_jeep_purchased_active_std,
                        nvehicles_brand_ram_purchased_active_std,nvehicles_purchased_std,
                        perc_total_spend_customer_paid_purchased_active_std,perc_total_spend_replacing_parts_purchased_active_std,
                        nVins_disposed_std,is_segment_c_std, mileage_recorded_from_last_service_for_active_owned_std
                        from tmc_ppo_ADS_MVP2_std_20200914
                        where is_retail = 1""")
    retail_defection_score = td.read_td(retail_defection_score_script, engine)
    return retail_defection_train, retail_defection_score



def lease_defection_data():
    lease_defection_train_script = ("""select did_churn_lease,nemails_opened_last_1yr_std,interest_present_in_hh_in_usa_travel,
                                    interest_present_in_hh_in_auto_work,days_since_last_service_leased_active_std,perc_total_spend_customer_paid_leased_active_std,
                                    perc_total_spend_replacing_parts_leased_active_std,navgdaysbetween_services_forlease_new_std,
                                    nleads_past90days,avg_service_downtime_leased_active_std,nvehicles_leased_active_std,
                                    days_to_lease_end_std,nrecalls_leased_active_std,days_since_first_recall_leased_active_std,
                                    is_leased_active_recall_open_std, is_leased_active_recall_closed_std, is_leased_active_recall_open_sseo_std,
                                    is_recall_type_safety_lease_std,is_recall_type_california_emission_lease_std,
                                    is_recall_type_customer_satisfaction_lease_std,is_recall_type_federal_emission_lease_std,
                                    days_between_recalls_leased_active_std,change_in_avg_snowfall_lease_std,
                                    avg_nservices_leased_active_std,days_between_leased_disposals_std,days_since_last_leased_disposal_std,
                                    nvehicles_leased_std from tmc_ppo_ADS_MVP2_lease_std_20190930""")
    lease_defection_train = td.read_td(lease_defection_train_script,engine)
    lease_defection_score_script = ("""select i_consmr,nemails_opened_last_1yr_std,interest_present_in_hh_in_usa_travel,
                                    interest_present_in_hh_in_auto_work,days_since_last_service_leased_active_std,
                                    perc_total_spend_customer_paid_leased_active_std,perc_total_spend_replacing_parts_leased_active_std,
                                    navgdaysbetween_services_forlease_new_std,nleads_past90days,avg_service_downtime_leased_active_std,
                                    nvehicles_leased_active_std,days_to_lease_end_std,nrecalls_leased_active_std,
                                    days_since_first_recall_leased_active_std,is_leased_active_recall_open_std, 
                                    is_leased_active_recall_closed_std, is_leased_active_recall_open_sseo_std,
                                    is_recall_type_safety_lease_std,is_recall_type_california_emission_lease_std,
                                    is_recall_type_customer_satisfaction_lease_std,is_recall_type_federal_emission_lease_std,
                                    days_between_recalls_leased_active_std,change_in_avg_snowfall_lease_std,
                                    avg_nservices_leased_active_std,days_between_leased_disposals_std,days_since_last_leased_disposal_std,
                                    nvehicles_leased_std from tmc_ppo_ADS_MVP2_std_20200914
                                    where is_retail = 0""")
    lease_defection_score = td.read_td(lease_defection_score_script, engine)
    return lease_defection_train, lease_defection_score 

def VA_Results():
    va_raw = ("""select * from tmc_product_affn_20200914_output""")
    va = td.read_td(va_raw,engine)
    return va

def PPO_Score():
    ppo_raw = ("""select * from tmc_ppo_2020_score_v2""")
    ppo = td.read_td(ppo_raw,engine)
    return ppo

def ppo_ads():
    ads_raw = ("""select * from tmc_ppo_BNSO_20200914_with_hh_id""")
    ads = td.read_td(ads_raw,engine)
    return ads
#%% Reducing DataFrame Size

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    for col in props.columns:
        if (props[col].dtype != object) & (props[col].isnull().values.any()==False):  # Exclude strings        
            # make variables for Int, max and min
            mx = props[col].max()
            mn = props[col].min()

            # Make Integer/unsigned Integer datatypes
            if mn >= 0:
                if mx < 255:
                    props[col] = props[col].astype(np.uint8)
                elif mx < 65535:
                    props[col] = props[col].astype(np.uint16)
                elif mx < 4294967295:
                    props[col] = props[col].astype(np.uint32)
                else:
                    props[col] = props[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    props[col] = props[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    props[col] = props[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    props[col] = props[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    props[col] = props[col].astype(np.int64)    
        
        # Make float datatypes 32 bit
        else:
            pass            
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props


#%% Data Pre-Processing

def data_prep_retail_inmarket(df):
    retail_inmarket_data = df.copy()
    retail_inmarket_data[['ndayssince_lastservice_forpurchase_new','ndayssince_last_purchase_new']] = retail_inmarket_data[['ndayssince_lastservice_forpurchase_new','ndayssince_last_purchase_new']].fillna(0)
    retail_inmarket_data['is_service_flag'] = np.where(retail_inmarket_data['ndayssince_lastservice_forpurchase_new']==0,1,0)
    retail_inmarket_data['avg_service_downtime_for_owned_new'] = retail_inmarket_data['avg_service_downtime_for_owned_new'].fillna(0)
    retail_inmarket_data['is_ram'] = np.where(retail_inmarket_data['ndayssince_last_purchase_new']== retail_inmarket_data['ndayssince_lastpurchase_brand_ram'],1,0 )
    retail_inmarket_data['is_dodge'] = np.where(retail_inmarket_data['ndayssince_last_purchase_new']== retail_inmarket_data['ndayssince_lastpurchase_brand_dodge'],1,0 )
    retail_inmarket_data['is_jeep'] = np.where(retail_inmarket_data['ndayssince_last_purchase_new']== retail_inmarket_data['ndayssince_lastpurchase_brand_jeep'],1,0 )
    retail_inmarket_data = retail_inmarket_data.drop(['ndayssince_lastpurchase_brand_ram','ndayssince_lastpurchase_brand_dodge','ndayssince_lastpurchase_brand_jeep'], axis =1)
    retail_inmarket_data = retail_inmarket_data.reset_index(drop = True)
    retail_inmarket_data.fillna(0)
    return retail_inmarket_data    
    
def data_prep_lease_inmarket(df):
    lease_inmarket_data = df.copy()
    lease_inmarket_data[['ndayssince_lastservice_forlease_new','navgdaysbetween_services_forlease_new_lt','avg_service_downtime_for_leased_new']] = lease_inmarket_data[['ndayssince_lastservice_forlease_new','navgdaysbetween_services_forlease_new_lt','avg_service_downtime_for_leased_new']].fillna(0)
    lease_inmarket_data['is_service_flag'] = np.where(lease_inmarket_data['ndayssince_lastservice_forlease_new']==0,1,0)
    lease_inmarket_data.fillna(0)
    return lease_inmarket_data  

    
def data_prep_retail_defection(df):
    retail_defection_data = df.copy()
    retail_defection_data[['days_since_last_service_purchased_active_std','avg_service_downtime_purchased_active_std','days_since_first_recall_purchased_active_std','days_between_recalls_purchased_active_std','change_in_avg_snowfall_retail_std','navgdaysbetween_services_forretail_new_std','mileage_recorded_from_last_service_for_active_owned_std']] = retail_defection_data[['days_since_last_service_purchased_active_std','avg_service_downtime_purchased_active_std','days_since_first_recall_purchased_active_std','days_between_recalls_purchased_active_std','change_in_avg_snowfall_retail_std','navgdaysbetween_services_forretail_new_std','mileage_recorded_from_last_service_for_active_owned_std']].fillna(0)
    retail_defection_data['one_service_done'] = np.where(retail_defection_data['navgdaysbetween_services_forretail_new_std']==0,1,0)
    cols_to_drop_r = ['avg_nservices_purchased_active_std','is_retail_active_recall_open_std',
                'is_recall_type_federal_emission_retail_std','is_recall_type_safety_retail_std',
                'is_recall_type_customer_satisfaction_retail_std','days_between_purchased_disposals_std',
                'days_between_recalls_purchased_active_std','days_since_first_recall_purchased_active_std',
                'is_recall_type_california_emission_retail_std','perc_total_spend_replacing_parts_purchased_active_std',
                'nvehicles_purchased_std','interest_present_in_hh_in_auto_work',
                'interest_present_in_hh_in_usa_travel','days_since_last_purchased_disposal_std',
                'nemails_opened_last_1yr_std','nLeads_past90Days','nvehicles_purchased_active_std']
    retail_defection_data = retail_defection_data.drop(cols_to_drop_r, axis =1)
    retail_defection_data.fillna(0)
    return retail_defection_data    


def data_prep_lease_defection(df):
    lease_defection_data = df.copy()
    lease_defection_data[['days_since_last_service_leased_active_std','avg_service_downtime_leased_active_std','navgdaysbetween_services_forlease_new_std','change_in_avg_snowfall_lease_std']] = lease_defection_data[['days_since_last_service_leased_active_std','avg_service_downtime_leased_active_std','navgdaysbetween_services_forlease_new_std','change_in_avg_snowfall_lease_std']].fillna(0)
    lease_defection_data['is_service_flag'] = np.where(lease_defection_data['days_since_last_service_leased_active_std']==0,1,0)
    cols_to_drop_l = ['days_since_first_recall_leased_active_std','days_between_recalls_leased_active_std',
                'is_recall_type_california_emission_lease_std','is_leased_active_recall_open_std',
                'is_leased_active_recall_closed_std','is_leased_active_recall_open_sseo_std',
                'is_recall_type_safety_lease_std','is_recall_type_customer_satisfaction_lease_std',
                'is_recall_type_federal_emission_lease_std','navgdaysbetween_services_forlease_new_std',
                'nvehicles_leased_std','nleads_past90days','days_between_leased_disposals_std','days_since_last_leased_disposal_std',
                'interest_present_in_hh_in_usa_travel','interest_present_in_hh_in_auto_work',
                'perc_total_spend_replacing_parts_leased_active_std']
    lease_defection_data = lease_defection_data.drop(cols_to_drop_l, axis =1)
    lease_defection_data.fillna(0)
    return lease_defection_data

#%% Outlier Treatment 

def outlier_lease_inmarket(df):
    lease_inmarket_outlier = df.copy()
    lease_inmarket_outlier['nemails_opened_last_1yr'] = np.where((lease_inmarket_outlier['nemails_opened_last_1yr']>lease_inmarket_outlier['nemails_opened_last_1yr'].quantile(0.99)), lease_inmarket_outlier['nemails_opened_last_1yr'].quantile(0.99),lease_inmarket_outlier['nemails_opened_last_1yr'])
    lease_inmarket_outlier['nvehiclesleased_new_lt'] = np.where((lease_inmarket_outlier['nvehiclesleased_new_lt']>lease_inmarket_outlier['nvehiclesleased_new_lt'].quantile(0.99)), lease_inmarket_outlier['nvehiclesleased_new_lt'].quantile(0.99),lease_inmarket_outlier['nvehiclesleased_new_lt'])
    lease_inmarket_outlier['ndayssince_lastservice_forlease_new'] = np.where((lease_inmarket_outlier['ndayssince_lastservice_forlease_new']>lease_inmarket_outlier['ndayssince_lastservice_forlease_new'].quantile(0.99)), lease_inmarket_outlier['ndayssince_lastservice_forlease_new'].quantile(0.99),lease_inmarket_outlier['ndayssince_lastservice_forlease_new'])
    lease_inmarket_outlier['spend_onservice_forleased_new_last_5yr'] = np.where((lease_inmarket_outlier['spend_onservice_forleased_new_last_5yr']>lease_inmarket_outlier['spend_onservice_forleased_new_last_5yr'].quantile(0.99)), lease_inmarket_outlier['spend_onservice_forleased_new_last_5yr'].quantile(0.99),lease_inmarket_outlier['spend_onservice_forleased_new_last_5yr'])
    lease_inmarket_outlier['nleads_past90days'] = np.where((lease_inmarket_outlier['nleads_past90days']>lease_inmarket_outlier['nleads_past90days'].quantile(0.99)), lease_inmarket_outlier['nleads_past90days'].quantile(0.99),lease_inmarket_outlier['nleads_past90days'])
    lease_inmarket_outlier['totalspend_onservice_forleased_new_last_5yr'] = np.where((lease_inmarket_outlier['totalspend_onservice_forleased_new_last_5yr']>lease_inmarket_outlier['totalspend_onservice_forleased_new_last_5yr'].quantile(0.99)), lease_inmarket_outlier['totalspend_onservice_forleased_new_last_5yr'].quantile(0.99),lease_inmarket_outlier['totalspend_onservice_forleased_new_last_5yr'])
    lease_inmarket_outlier['avg_service_downtime_for_leased_new'] = np.where((lease_inmarket_outlier['avg_service_downtime_for_leased_new']>lease_inmarket_outlier['avg_service_downtime_for_leased_new'].quantile(0.99)), lease_inmarket_outlier['avg_service_downtime_for_leased_new'].quantile(0.99),lease_inmarket_outlier['avg_service_downtime_for_leased_new'])
    lease_inmarket_outlier['navgdaysbetween_services_forlease_new_lt'] = np.where((lease_inmarket_outlier['navgdaysbetween_services_forlease_new_lt']>lease_inmarket_outlier['navgdaysbetween_services_forlease_new_lt'].quantile(0.99)), lease_inmarket_outlier['navgdaysbetween_services_forlease_new_lt'].quantile(0.99),lease_inmarket_outlier['navgdaysbetween_services_forlease_new_lt'])
    return lease_inmarket_outlier

def outlier_retail_defection(df):
    retail_defection_outlier = df.copy()
    retail_defection_outlier['days_since_last_service_purchased_active_std'] = np.where((retail_defection_outlier['days_since_last_service_purchased_active_std']>retail_defection_outlier['days_since_last_service_purchased_active_std'].quantile(0.99)), retail_defection_outlier['days_since_last_service_purchased_active_std'].quantile(0.99),retail_defection_outlier['days_since_last_service_purchased_active_std'])
    retail_defection_outlier['avg_service_downtime_purchased_active_std'] = np.where((retail_defection_outlier['avg_service_downtime_purchased_active_std']>retail_defection_outlier['avg_service_downtime_purchased_active_std'].quantile(0.99)), retail_defection_outlier['avg_service_downtime_purchased_active_std'].quantile(0.99),retail_defection_outlier['avg_service_downtime_purchased_active_std'])
    retail_defection_outlier['nRecalls_purchased_active_std'] = np.where((retail_defection_outlier['nRecalls_purchased_active_std']>retail_defection_outlier['nRecalls_purchased_active_std'].quantile(0.99)), retail_defection_outlier['nRecalls_purchased_active_std'].quantile(0.99),retail_defection_outlier['nRecalls_purchased_active_std'])
    retail_defection_outlier['change_in_avg_snowfall_retail_std'] = np.where((retail_defection_outlier['change_in_avg_snowfall_retail_std']>retail_defection_outlier['change_in_avg_snowfall_retail_std'].quantile(0.99)), retail_defection_outlier['change_in_avg_snowfall_retail_std'].quantile(0.99),retail_defection_outlier['change_in_avg_snowfall_retail_std'])
    retail_defection_outlier['navgdaysbetween_services_forretail_new_std'] = np.where((retail_defection_outlier['navgdaysbetween_services_forretail_new_std']>retail_defection_outlier['navgdaysbetween_services_forretail_new_std'].quantile(0.99)), retail_defection_outlier['navgdaysbetween_services_forretail_new_std'].quantile(0.99),retail_defection_outlier['navgdaysbetween_services_forretail_new_std'])
    retail_defection_outlier['nvehicles_brand_dodge_purchased_active_std'] = np.where((retail_defection_outlier['nvehicles_brand_dodge_purchased_active_std']>retail_defection_outlier['nvehicles_brand_dodge_purchased_active_std'].quantile(0.99)), retail_defection_outlier['nvehicles_brand_dodge_purchased_active_std'].quantile(0.99),retail_defection_outlier['nvehicles_brand_dodge_purchased_active_std'])
    retail_defection_outlier['nvehicles_brand_jeep_purchased_active_std'] = np.where((retail_defection_outlier['nvehicles_brand_jeep_purchased_active_std']>retail_defection_outlier['nvehicles_brand_jeep_purchased_active_std'].quantile(0.99)), retail_defection_outlier['nvehicles_brand_jeep_purchased_active_std'].quantile(0.99),retail_defection_outlier['nvehicles_brand_jeep_purchased_active_std'])
    retail_defection_outlier['nvehicles_brand_ram_purchased_active_std'] = np.where((retail_defection_outlier['nvehicles_brand_ram_purchased_active_std']>retail_defection_outlier['nvehicles_brand_ram_purchased_active_std'].quantile(0.99)), retail_defection_outlier['nvehicles_brand_ram_purchased_active_std'].quantile(0.99),retail_defection_outlier['nvehicles_brand_ram_purchased_active_std'])
    retail_defection_outlier['perc_total_spend_customer_paid_purchased_active_std'] = np.where((retail_defection_outlier['perc_total_spend_customer_paid_purchased_active_std']>retail_defection_outlier['perc_total_spend_customer_paid_purchased_active_std'].quantile(0.99)), retail_defection_outlier['perc_total_spend_customer_paid_purchased_active_std'].quantile(0.99),retail_defection_outlier['perc_total_spend_customer_paid_purchased_active_std'])
    retail_defection_outlier['nVins_disposed_std'] = np.where((retail_defection_outlier['nVins_disposed_std']>retail_defection_outlier['nVins_disposed_std'].quantile(0.99)), retail_defection_outlier['nVins_disposed_std'].quantile(0.99),retail_defection_outlier['nVins_disposed_std'])
    retail_defection_outlier['mileage_recorded_from_last_service_for_active_owned_std'] = np.where((retail_defection_outlier['mileage_recorded_from_last_service_for_active_owned_std']>retail_defection_outlier['mileage_recorded_from_last_service_for_active_owned_std'].quantile(0.99)), retail_defection_outlier['mileage_recorded_from_last_service_for_active_owned_std'].quantile(0.99),retail_defection_outlier['mileage_recorded_from_last_service_for_active_owned_std'])
    return retail_defection_outlier

def outlier_lease_defection(df):
    lease_defection_outlier = df.copy()
    lease_defection_outlier['nemails_opened_last_1yr_std'] = np.where((lease_defection_outlier['nemails_opened_last_1yr_std']>lease_defection_outlier['nemails_opened_last_1yr_std'].quantile(0.99)), lease_defection_outlier['nemails_opened_last_1yr_std'].quantile(0.99),lease_defection_outlier['nemails_opened_last_1yr_std'])
    lease_defection_outlier['days_since_last_service_leased_active_std'] = np.where((lease_defection_outlier['days_since_last_service_leased_active_std']>lease_defection_outlier['days_since_last_service_leased_active_std'].quantile(0.99)), lease_defection_outlier['days_since_last_service_leased_active_std'].quantile(0.99),lease_defection_outlier['days_since_last_service_leased_active_std'])
    lease_defection_outlier['avg_service_downtime_leased_active_std'] = np.where((lease_defection_outlier['avg_service_downtime_leased_active_std']>lease_defection_outlier['avg_service_downtime_leased_active_std'].quantile(0.99)), lease_defection_outlier['avg_service_downtime_leased_active_std'].quantile(0.99),lease_defection_outlier['avg_service_downtime_leased_active_std'])
    lease_defection_outlier['nvehicles_leased_active_std'] = np.where((lease_defection_outlier['nvehicles_leased_active_std']>lease_defection_outlier['nvehicles_leased_active_std'].quantile(0.99)), lease_defection_outlier['nvehicles_leased_active_std'].quantile(0.99),lease_defection_outlier['nvehicles_leased_active_std'])
    lease_defection_outlier['nrecalls_leased_active_std'] = np.where((lease_defection_outlier['nrecalls_leased_active_std']>lease_defection_outlier['nrecalls_leased_active_std'].quantile(0.99)), lease_defection_outlier['nrecalls_leased_active_std'].quantile(0.99),lease_defection_outlier['nrecalls_leased_active_std'])
    lease_defection_outlier['change_in_avg_snowfall_lease_std'] = np.where((lease_defection_outlier['change_in_avg_snowfall_lease_std']>lease_defection_outlier['change_in_avg_snowfall_lease_std'].quantile(0.99)), lease_defection_outlier['change_in_avg_snowfall_lease_std'].quantile(0.99),lease_defection_outlier['change_in_avg_snowfall_lease_std'])
    return lease_defection_outlier


#%% Scaling

def scaling_retail_inmarket(df):
    scaling_retail_inmarket_data = df.copy()
    col_names_retail_inmarket = ['nemails_opened_last_1yr','nvehiclesowned_new_active_last_5yrs','nservicesdone_forpurchase_new_active_last_5yrs','ndayssince_lastservice_forpurchase_new',
       'spend_onservice_forowned_new_last_5yr', 'nLeads_past90Days','totalspend_onservice_forowned_new_last_5yr','avg_service_downtime_for_owned_new', 'ndayssince_last_purchase_new']
    features_retail_inmarket = scaling_retail_inmarket_data[col_names_retail_inmarket]
    scaler_retail_inmarket = StandardScaler().fit(features_retail_inmarket.values)
    features_retail_inmarket = scaler_retail_inmarket.transform(features_retail_inmarket.values)
    scaling_retail_inmarket_data[col_names_retail_inmarket] = features_retail_inmarket
    return scaling_retail_inmarket_data


def scaling_lease_inmarket(df):
    scaling_lease_inmarket_data = df.copy()
    col_names_lease_inmarket = ['nemails_opened_last_1yr','nvehiclesleased_new_lt','ndayssince_lastservice_forlease_new',
             'spend_onservice_forleased_new_last_5yr','nleads_past90days','totalspend_onservice_forleased_new_last_5yr','avg_service_downtime_for_leased_new',
             'navgdaysbetween_services_forlease_new_lt']
    features_lease_inmarket = scaling_lease_inmarket_data[col_names_lease_inmarket]
    scaler_lease_inmarket = StandardScaler().fit(features_lease_inmarket.values)
    features_lease_inmarket = scaler_lease_inmarket.transform(features_lease_inmarket.values)
    scaling_lease_inmarket_data[col_names_lease_inmarket] = features_lease_inmarket
    return scaling_lease_inmarket_data

def scaling_retail_defection(df):
    scaling_retail_defection_data = df.copy()
    col_names_retail_defection = ['avg_service_downtime_purchased_active_std','change_in_avg_snowfall_retail_std',
       'days_since_last_service_purchased_active_std','nRecalls_purchased_active_std', 'navgdaysbetween_services_forretail_new_std',
       'nvehicles_brand_dodge_purchased_active_std','nvehicles_brand_jeep_purchased_active_std',
       'nvehicles_brand_ram_purchased_active_std','perc_total_spend_customer_paid_purchased_active_std','nVins_disposed_std',
       'mileage_recorded_from_last_service_for_active_owned_std']
    features_retail_defection = scaling_retail_defection_data[col_names_retail_defection]
    scaler_retail_defection = StandardScaler().fit(features_retail_defection.values)
    features_retail_defection = scaler_retail_defection.transform(features_retail_defection.values)
    scaling_retail_defection_data[col_names_retail_defection] = features_retail_defection
    return scaling_retail_defection_data

def scaling_lease_defection(df):
    scaling_lease_defection_data = df.copy()
    col_names_lease_defection = ['nemails_opened_last_1yr_std','days_since_last_service_leased_active_std',
       'perc_total_spend_customer_paid_leased_active_std','avg_service_downtime_leased_active_std', 'nvehicles_leased_active_std',
       'days_to_lease_end_std', 'nrecalls_leased_active_std','change_in_avg_snowfall_lease_std', 'avg_nservices_leased_active_std']
    features_lease_defection = scaling_lease_defection_data[col_names_lease_defection]
    scaler_lease_defection = StandardScaler().fit(features_lease_defection.values)
    features_lease_defection = scaler_lease_defection.transform(features_lease_defection.values)
    scaling_lease_defection_data[col_names_lease_defection] = features_lease_defection
    return scaling_lease_defection_data    


#%% Smoting

def SamplingAndSplit(df, dept):
    over_sample_data = df.copy()
    X = np.array(over_sample_data.drop(columns = dept,axis=1))
    y = np.array(over_sample_data[dept])
    
    # Train- Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2, stratify = y)
    
    # Smote
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
     
    sm = SMOTE(random_state = 2) 
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
      
    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))   
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))
    return  X_train_res, y_train_res, X_test, y_test  



#%% Logit Model to get beta and significant features

def RunLogit(ytrain, xtrain, scaleTrain, dept):
    logit_model = sm.Logit(ytrain,xtrain)
    result = logit_model.fit()
    
    results_df = pd.DataFrame({"coeff":result.params,"pvals":result.pvalues})
    results_df['variable'] = scaleTrain.drop(columns = dept,axis=1).columns.tolist()
    
    final = []
    rem_feature = []
    
    while len(results_df[results_df['pvals']>=0.05]['pvals'])>0:
        
        a = []
        a = results_df[results_df['pvals']>=0.05]['pvals'].index.tolist()
        Xt = np.delete(xtrain,a,axis=1)
        
        logit_model = sm.Logit(ytrain,Xt)
        result = logit_model.fit()
        xtrain = Xt
        final.append(a)
        
        res_df = pd.DataFrame()
        res_df = pd.DataFrame({"coeff":result.params,"pvals":result.pvalues})
        res_df['variable'] = results_df['variable'].drop(a).tolist()
        results_df = res_df 
        
        
    rem_feature = scaleTrain.drop(columns = dept,axis=1).columns[final].tolist()
    print("Insignificant Features are {}".format(rem_feature))
    return final, rem_feature, results_df
    
    
#%% Logistics Model

def BuildlogModel(xtrn, ytrn, xtst, ytst, remove):
     
    clf = LogisticRegression()
    grid_value = {'penalty':['l2','elasticnet'],
              'C':[0.1,1,50,100,200],
              'solver':['saga'],
              'max_iter': [300]}
    grid_clf_acc = GridSearchCV(clf,param_grid=grid_value,scoring='recall',n_jobs=-1)
    xtrn_new = np.delete(xtrn,remove,axis=1)
    xtst_new = np.delete(xtst,remove,axis=1)
    grid_clf_acc.fit(xtrn_new,ytrn)
    
    y_pred = grid_clf_acc.predict(xtst_new)
    y_train_pred = grid_clf_acc.predict(xtrn_new)
    
    print("Train Accuracy is :",metrics.accuracy_score(ytrn, y_train_pred))
    print("Test Accuracy is :",metrics.accuracy_score(ytst, y_pred))
    
    print("Train ")
    print(classification_report(ytrn, y_train_pred)) 
    
    print("\nTest ")
    print(classification_report(ytst, y_pred)) 
    return grid_clf_acc

#%% Scoring Model

def ScorelogModel(xscore ,mod ,remove, traindata, dept, scoring_name, scoring_prob_name):
    
    col = traindata.drop([dept],axis=1).columns.tolist()
    xscore_new = xscore[col]
    xscore_new = np.array(xscore_new)
    xscore_new = np.delete(xscore_new,remove,axis=1)
    
    val_pred = mod.predict(np.array(xscore_new))
    val_prob = pd.DataFrame(mod.predict_proba(np.array(xscore_new)))
    val_prob = val_prob.add_prefix(scoring_prob_name)
    
    output = xscore[['i_consmr']]
    output[scoring_name] = val_pred
    output = pd.concat([output,val_prob], axis = 1)  
    return output 

#%% Decile Score

def DefectionDecileScore(df, prob_filter, name_decile):
    dec = df.copy()
    bin_labels_10 = ['D1', 'D2', 'D3', 'D4', 'D5','D6','D7','D8','D9','D10'] 
    q = [x for x in range(0,len(dec),math.ceil(len(dec)/10))]
    q.append(len(dec))
    
    dec = dec.sort_values(prob_filter,ascending= False).reset_index(drop=True)
    dec[name_decile] = pd.cut(dec.index, q , labels = bin_labels_10)
    dec[name_decile] = dec[name_decile].fillna('D1')

    dec_summary = dec.groupby([name_decile]).agg({prob_filter:['min','max']}).reset_index()
    dec_summary.columns = dec_summary.columns.droplevel(1)
    
    count = dec[name_decile].value_counts().reset_index()
    count.columns = [name_decile,'count']
    
    dec_summary = dec_summary.merge(count, on=name_decile, how='left')
    return dec, dec_summary

def InmarketDecileScore(df, prob_filter, name_decile):
    dec = df.copy()
    bin_labels_10 = ['I1', 'I2', 'I3', 'I4', 'I5','I6','I7','I8','I9','I10'] 
    q = [x for x in range(0,len(dec),math.ceil(len(dec)/10))]
    q.append(len(dec))
    
    dec = dec.sort_values(prob_filter,ascending= False).reset_index(drop=True)
    dec[name_decile] = pd.cut(dec.index, q , labels = bin_labels_10)
    dec[name_decile] = dec[name_decile].fillna('I1')

    dec_summary = dec.groupby([name_decile]).agg({prob_filter:['min','max']}).reset_index()
    dec_summary.columns = dec_summary.columns.droplevel(1)
    
    count = dec[name_decile].value_counts().reset_index()
    count.columns = [name_decile,'count']
    
    dec_summary = dec_summary.merge(count, on=name_decile, how='left')
    return dec, dec_summary
    
#%%
    
