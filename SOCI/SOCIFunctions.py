

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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import math
import pdb
print("\n All Libraries import done")

#%% Building Conncetion
With Databases.
#%% Loading Tables
#----------------------------------------------------------------VISIT---------------------------------------------------------
def BNSOretail():
    #added new table name "tmc_mopar_soci_ads_20200814"
    #added new table name "tmc_mopar_soci_ads_20200914"
    BNSOretail_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'new' and lower(trim(purchase_type)) = 'retail' """)
    BNSOretail_train = td.read_td(BNSOretail_train_script,engine)
    BNSOretail_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'new' and lower(trim(purchase_type)) = 'retail' """)
    BNSOretail_score = td.read_td(BNSOretail_score_script,engine)
    return BNSOretail_train,BNSOretail_score

def BNSOlease():
    #added new table name "tmc_mopar_soci_ads_20200814"
    #added new table name "tmc_mopar_soci_ads_20200914"
    BNSOlease_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'new' and lower(trim(purchase_type)) = 'lease' AND MAILABLE = 1 """)
    BNSOlease_train = td.read_td(BNSOlease_train_script,engine)
    BNSOlease_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'new' and lower(trim(purchase_type)) = 'lease'""")
    BNSOlease_score = td.read_td(BNSOlease_score_script,engine)
    return BNSOlease_train,BNSOlease_score

def BUSO_withservice():
    #added new table name "tmc_mopar_soci_ads_20200814"
    #added new table name "tmc_mopar_soci_ads_20200914"
    BUSO_ws_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'used' and ndayssince_lastservice is not NULL """)
    BUSO_ws_train = td.read_td(BUSO_ws_train_script,engine)
    BNSO_ws_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'used' and ndayssince_lastservice is not NULL """)
    BNSO_ws_score = td.read_td(BNSO_ws_score_script,engine)
    return BUSO_ws_train,BNSO_ws_score


def BUSO_withoutservice():
     #added new table name "tmc_mopar_soci_ads_20200814"
    #added new table name "tmc_mopar_soci_ads_20200914"
    BUSO_wo_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'used' and ndayssince_lastservice is  NULL AND MAILABLE = 1 """)
    BUSO_wo_train = td.read_td(BUSO_wo_train_script,engine)
    BNSO_wo_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'used' and ndayssince_lastservice is  NULL """)
    BNSO_wo_score = td.read_td(BNSO_wo_score_script,engine)
    return BUSO_wo_train,BNSO_wo_score
#-----------------------------------------------------------SPEND---------------------------------------------------------
def BNSO_spend():
    BNSO_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'new' and dep_servicevisit = 1  """)
    BNSO_train = td.read_td(BNSO_train_script,engine)
    BNSO_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'new'  """)
    BNSO_score = td.read_td(BNSO_score_script,engine)
    return BNSO_train,BNSO_score
    
def BUSO_ws_spend():
    BNSO_ws_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'used' and dep_servicevisit = 1 and ndayssince_lastservice is NOT NULL""")
    BNSO_ws_train = td.read_td(BNSO_ws_train_script,engine)
    BNSO_ws_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'used'  and ndayssince_lastservice is NOT NULL""")
    BNSO_ws_score = td.read_td(BNSO_ws_score_script,engine)
    return BNSO_ws_train,BNSO_ws_score

def BUSO_wo_spend():
    BUSO_wo_train_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20200921 where lower(trim(new_used_flag)) = 'used' and dep_servicevisit = 1 and ndayssince_lastservice is NULL""")
    BUSO_wo_train = td.read_td(BUSO_wo_train_script,engine)
    BUSO_wo_score_script = (""" select * from sandbox_data_science.tmc_mopar_soci_ads_20201022 where lower(trim(new_used_flag)) = 'used'  and ndayssince_lastservice is NULL""")
    BUSO_wo_score = td.read_td(BUSO_wo_score_script,engine)
    return BUSO_wo_train,BUSO_wo_score


#%% Reduce Memory
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


#%%  DATA PREPARATION

##Service Model data preparation

columns_to_drop_retail_visit = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag',
       'purchase_type', 'make', 'nameplate', 'vehicle_trim', 'dep_total_spent', 'days_to_lease_end','change_in_avg_snowfall','distance_serv_last','time','mailable','dealer_assigned','proportion_spend_parts_total']
    
def BNSO_retail_dataprep(df,columns_info = columns_to_drop_retail_visit):
    # pdb.set_trace()
    data_retail = df.copy()
    data_retail = data_retail.drop(columns_info,axis=1)
    # data_retail['proportion_spend_parts_missing'] = np.where(data_retail['proportion_spend_parts_total']=='NaN',1,0)
    # data_retail['proportion_spend_parts_total'] = data_retail['proportion_spend_parts_total'].astype(float)
    # data_retail['proportion_spend_parts_total']= data_retail['proportion_spend_parts_total'].replace([np.inf, -np.inf], np.nan)
    data_retail['non_service'] = np.where(data_retail['ndayssince_lastservice'].isnull()==True,1,0)
    data_retail['avg_service_missing'] = np.where(data_retail['navgdaysbetween_services'].isnull()==True,1,0)
    data_retail = pd.get_dummies(data=data_retail, columns=['fuel_type','engine_type','car_segment','fam_comp_code'])
    data_retail = data_retail.fillna(0)
    data_retail.clip(lower=0,inplace = True)
    return data_retail

### ServiceModel Lease data preparation
columns_to_drop_lease_visit = [ 'make', 'nameplate','proportion_spend_parts_total', 'vehicle_trim', 'dep_total_spent','fuel_type', 'change_in_avg_snowfall','time','age_of_vehicle_in_years','avg_service_downtime','demo_gender_female','demo_gender_male','demo_gender_unknown','demog_age_18_to_25','demog_age_26_to_35','demog_age_36_to_45','demog_age_46_to_55','demog_age_56_to_65','demog_age_above_65','demog_age_not_captured',
                               'income_band_135000_160000','income_band_45000_55000','income_band_95000_115000',
                               'service_type_tire_rotation','income_band_75000_95000','recall_num',
                               'car_segment','demog_age_46_to_55','demog_age_36_to_45',
                               'demog_age_56_to_65','demo_gender_male','service_type_brake','service_type_charging_test']

def BNSO_lease_dataprep(df,columns_info = columns_to_drop_lease_visit):
    data_lease = df.copy()
    
    df_1 = data_lease[data_lease['last_milg_recorded'].isnull() == True]
    df_1.drop('last_milg_recorded',axis=1,inplace=True)
    df_2 = data_lease[data_lease['last_milg_recorded'].isnull() == False]
    df_3 = df_2[['age_of_vehicle_in_years','last_milg_recorded']].groupby(['age_of_vehicle_in_years']).median()
    df_1 = df_1.merge(df_3,how='left',left_on = 'age_of_vehicle_in_years',right_on = 'age_of_vehicle_in_years')
    data_lease = pd.concat([df_2,df_1],axis=0,ignore_index=True)
    
    data_lease = data_lease.drop(columns_info,axis=1)
    
    data_lease[['distance_serv_last']].fillna(data_lease[['distance_serv_last']].median(),inplace=True)
    data_lease[['ndayssince_lastservice']].fillna(data_lease[['ndayssince_lastservice']].median(),inplace=True)
    
    
    data_lease['avg_service_missing'] = np.where(data_lease['navgdaysbetween_services'].isnull()==True,1,0)
    data_lease = pd.get_dummies(data=data_lease, columns=['engine_type','fam_comp_code'])
    data_lease = data_lease.fillna(0)
    return data_lease

### BUSO wo service
    
columns_to_drop_buso_wo_service = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag',
       'purchase_type', 'make', 'nameplate', 'vehicle_trim', 'dep_total_spent','proportion_spend_parts_total', 'days_to_lease_end',
       'change_in_avg_snowfall','distance_serv_last','warranty_flag','fam_comp_code','time','avg_service_downtime','last_milg_recorded',
       'ndayssince_lastservice','fuel_type','engine_type','navgdaysbetween_services',"mailable",'dealer_assigned']

def BUSO_wo_service_dataprep(df,columns_info = columns_to_drop_buso_wo_service):
    data_buso_wo_service = df.copy()
    data_buso_wo_service = data_buso_wo_service.drop(columns_info,axis=1)
    data_buso_wo_service = pd.get_dummies(data=data_buso_wo_service, columns=['car_segment'])
    data_buso_wo_service = data_buso_wo_service.fillna(0)
    data_buso_wo_service.clip(lower=0,inplace = True)
    return data_buso_wo_service

### BUSO With Serrvice
columns_to_drop_buso_service = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag',
       'purchase_type', 'make', 'nameplate', 'vehicle_trim', 'dep_total_spent','proportion_spend_parts_total', 'days_to_lease_end',
       'change_in_avg_snowfall','distance_serv_last','warranty_flag','fam_comp_code','time',"mailable",'dealer_assigned']

def BUSO_service_dataprep(df,columns_info = columns_to_drop_buso_service):
    data_buso_service = df.copy()
    data_buso_service = data_buso_service.drop(columns_info,axis=1)
    data_buso_service['non_service'] = np.where(data_buso_service['ndayssince_lastservice'].isnull()==True,1,0)
    data_buso_service['avg_service_missing'] = np.where(data_buso_service['navgdaysbetween_services'].isnull()==True,1,0)
    data_buso_service = pd.get_dummies(data=data_buso_service, columns=['fuel_type','engine_type','car_segment'])
    data_buso_service = data_buso_service.fillna(0)
    data_buso_service.clip(lower=0,inplace = True)
    return data_buso_service


#%% Outlier Treatment
def outlier_generic(df,dept=None,max_flag=True, ty = None):
    """use ty=="tr" for train and give value in dept ;
       use ty==None for score dept is not required"""       
    df_generic = df.copy()    
    if ty=="tr":
        df_generic = df_generic.drop(columns = dept,axis=1)
    elif dept != None:
        df_generic = df_generic.drop(columns = dept,axis=1)     
    if max_flag:
        col_list = df_generic.describe().T.loc[df_generic.describe().T["max"]>1].index
    else:
        col_list = df_generic.describe().columns   
    for num_col in col_list:
        df_generic[num_col] = np.where((df_generic[num_col]>df_generic[num_col].quantile(0.99)), df_generic[num_col].quantile(0.99),df_generic[num_col])       
    if ty=="tr" or dept != None:
        df_generic = pd.concat([df[[dept]],df_generic],axis =1)            
    return df_generic, col_list

#%% SPEND DATA PREPARATION
    
#--------------------------------------------------------SPEND PREP--------------------------------------------------------
columns_to_drop_retail_spend = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag',
        'nameplate', 'vehicle_trim','dep_servicevisit', 'proportion_spend_parts_total', 
       'change_in_avg_snowfall','distance_serv_last','time','days_to_lease_end','dealer_assigned']  


def spend_bucket(df):
    if df['dep_total_spent']>=1000 :
        return 7
    elif (df['dep_total_spent']>=500) & (df['dep_total_spent']<1000):
        return 6
    elif (df['dep_total_spent']>=400) & (df['dep_total_spent']<500):
        return 5
    elif (df['dep_total_spent']>=300) & (df['dep_total_spent']<400):
        return 4
    elif (df['dep_total_spent']>=200) & (df['dep_total_spent']<300):
        return 3
    elif (df['dep_total_spent']>=100) & (df['dep_total_spent']<200):
        return 2   
    elif (df['dep_total_spent']>=0) & (df['dep_total_spent']<100):
        return 1 
    
def BNSO_spend_dataprep(df,columns_info = columns_to_drop_retail_spend, ty = None):
    data_spend = df.copy()      
    data_spend = data_spend.drop(columns_info,axis=1)
    data_spend['non_service'] = np.where(data_spend['ndayssince_lastservice'].isnull()==True,1,0)
    data_spend['avg_service_missing'] = np.where(data_spend['navgdaysbetween_services'].isnull()==True,1,0)
    data_spend = pd.get_dummies(data=data_spend, columns=['purchase_type','make','fuel_type','engine_type','car_segment','fam_comp_code'])
    data_spend = data_spend.fillna(0)
    data_spend.clip(lower=0,inplace = True)
    
    if (ty =='tr'):
        data_spend['spend_group'] = data_spend.apply(spend_bucket, axis=1)
        data_spend = data_spend.drop('dep_total_spent',axis=1)
    else: 
        pass
    
    return data_spend

columns_to_drop_buso_wo_spend = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag',
        'nameplate', 'vehicle_trim','purchase_type','dep_servicevisit', 'proportion_spend_parts_total', 
       'avg_service_downtime','last_milg_recorded','ndayssince_lastservice',
       'fuel_type','engine_type','navgdaysbetween_services',
        'change_in_avg_snowfall','distance_serv_last','days_to_lease_end','warranty_flag','fam_comp_code','time','dealer_assigned']  

def BUSO_wo_spend_dataprep(df,columns_info = columns_to_drop_buso_wo_spend, ty = None):
    data_spend = df.copy()
    data_spend = data_spend.drop(columns_info,axis=1)
    data_spend = pd.get_dummies(data=data_spend, columns=['make','car_segment'])
    data_spend = data_spend.fillna(0)
    data_spend.clip(lower=0,inplace = True)
    
    if (ty =='tr'):
        data_spend['spend_group'] = data_spend.apply(spend_bucket, axis=1)
        data_spend = data_spend.drop('dep_total_spent',axis=1)
    else: 
        data_spend = data_spend.drop('dep_total_spent',axis=1)
        
    return data_spend

#%% Scaling Standardization 
def scaling_generic(df,df_score,l):
    scale_train = df.copy()
    scale_score = df_score.copy()
    
    df_generic_features = scale_train[l]
    df_generic_score = scale_score[l]
    
    generic_scaler = MinMaxScaler().fit(df_generic_features.values)
    generic_features = generic_scaler.transform(df_generic_features.values)
    generic_features_score  = generic_scaler.transform(df_generic_score.values)
    
    scale_train[l]=generic_features
    scale_score[l]=generic_features_score
    return scale_train,scale_score    


def DataCleaner(df):
    data_c = df.copy()
    col_remove = data_c.describe().T.loc[(data_c.describe().T["max"]==0) | (data_c.describe().T["std"]==0)].index
    data_c = data_c.drop(col_remove, axis =1)
    print("Columns removed are {}".format(col_remove))
    return data_c, col_remove

def DataCommon(df1,df2):
    com = df1.columns.intersection(df2.columns)
    df1 = df1.copy()[com]
    df2 = df2.copy()[com]
    return df1,df2


def uncorrelated_features(df, dept , threshold ):
    """
    Returns a subset of df columns with Pearson correlations
    below threshold.
    """
    data_un = df.copy()
    data_un = data_un.drop(columns = dept,axis=1)
    corr = data_un.corr().abs()
    keep = []
    for i in range(len(corr.iloc[:,0])):
        above = corr.iloc[:i,i]
        if len(keep) > 0: above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(corr.columns.values[i])
    output = data_un[keep]            
    output = pd.concat([output,df[dept]],axis =1)
    print("Columns Kept are {}".format(keep))    
    return output

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

#-------------------------------------------------------------SPEND------------------------------------------
def StratifySampling(df,dept):
    over_sample_data = df.copy()
    
    X = np.array(over_sample_data.drop(columns = dept,axis=1))
    y = np.array(over_sample_data[dept])
    
    # Train- Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2, stratify = y)
    return  X_train, y_train, X_test, y_test

#%% Significance Removal
    

def RunLogit(ytrain, xtrain, scaleTrain, dept):
    # pdb.set_trace()
    logit_model = sm.Logit(ytrain,xtrain)
    result = logit_model.fit()
    
    results_df = pd.DataFrame({"coeff":result.params,"pvals":result.pvalues})
    results_df['variable'] = scaleTrain.drop(columns = dept,axis=1).columns.tolist()
    
    final = []
    rem_feature = []
    flat_list = []
    
    while len(results_df[results_df['pvals']>=0.05]['pvals'])>0:
        
        a = []
        a = results_df[results_df['pvals']>=0.05]['pvals'].index.tolist()
        Xt = np.delete(xtrain,a,axis=1)
        
        logit_model = sm.Logit(ytrain,Xt)
        result = logit_model.fit()
        xtrain = Xt
        final.append(a)
        
    
        for sublist in final:
           for item in sublist:
               flat_list.append(item)
        
        res_df = pd.DataFrame()
        res_df = pd.DataFrame({"coeff":result.params,"pvals":result.pvalues})
        res_df['variable'] = results_df['variable'].drop(a).tolist()
        results_df = res_df 
        
    flat_list = list(set(flat_list))            
    rem_feature = scaleTrain.drop(columns = dept,axis=1).columns[flat_list].tolist()
    print("Insignificant Features are {}".format(rem_feature))
    return flat_list, rem_feature, results_df


#%%  Model Building

####Logistics Regression
    
def BuildlogModel(xtrn, ytrn, xtst, ytst, remove,typ=None):
    if typ == "LEASE":
        grid_clf_acc = LogisticRegression(C=10,max_iter=200,penalty='l2',solver ='saga', n_jobs=-1) #for BUSO Lease only
    else:
        grid_clf_acc = LogisticRegression(penalty='l2',solver ='saga', n_jobs=-1) #for all
        
    xtrn_new = np.delete(xtrn,remove,axis=1)
    xtst_new = np.delete(xtst,remove,axis=1)
    grid_clf_acc.fit(xtrn_new,ytrn)
    y_pred = grid_clf_acc.predict(xtst_new)
    y_train_pred = grid_clf_acc.predict(xtrn_new)
    # print("grid best para",grid_clf_acc.best_params_)
    print("Train Accuracy is :",metrics.accuracy_score(ytrn, y_train_pred))
    print("Test Accuracy is :",metrics.accuracy_score(ytst, y_pred))
    
    print("Train ")
    print(classification_report(ytrn, y_train_pred)) 
    
    print("\nTest ")
    print(classification_report(ytst, y_pred)) 
    return grid_clf_acc    



def BuildRandomModel(xtrn, ytrn, xtst, ytst, remove):

    xtrn_new = np.delete(xtrn,remove,axis=1)
    xtst_new = np.delete(xtst,remove,axis=1)    
    
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(xtrn_new,ytrn)    
    y_pred = clf.predict(xtst_new)
    y_train_pred = clf.predict(xtrn_new)
    
    print("Train Accuracy is :",metrics.accuracy_score(ytrn, y_train_pred))
    print("Test Accuracy is :",metrics.accuracy_score(ytst, y_pred))
    
    print("Train ")
    print(classification_report(ytrn, y_train_pred)) 
    
    print("\nTest ")
    print(classification_report(ytst, y_pred)) 
    return clf    



def ScorelogModel(xscore ,mod ,remove, traindata, dept, scoring_name, scoring_prob_name,mod_type_dic={"mod_type":"BNSO","mail_flag":1,"col_keep":[]}):
    
    if mod_type_dic["mod_type"]=="BNSO":
        col = traindata.drop([dept],axis=1).columns.tolist()
        xscore_new = xscore[col]
        xscore_new = np.array(xscore_new)
        xscore_new = np.delete(xscore_new,remove,axis=1)
        
        val_pred = mod.predict(np.array(xscore_new))
        val_prob = pd.DataFrame(mod.predict_proba(np.array(xscore_new)))
        val_prob = val_prob.add_prefix(scoring_prob_name)
        
        output = xscore[['dep_servicevisit']]
        output[scoring_name] = val_pred
        output = pd.concat([output,val_prob], axis = 1) 
        print(classification_report(output['dep_servicevisit'], output[scoring_name]))
        return output
    else:
        col = traindata.drop([dept],axis=1).columns.tolist()
        xscore_new = xscore[xscore["mailable"] == mod_type_dic["mail_flag"]].reset_index(drop=True)
        raw2 = xscore_new[mod_type_dic['col_keep']].copy()
        xscore_dep = xscore_new[['dep_servicevisit']]
        xscore_new = xscore_new[col]
        xscore_new = np.array(xscore_new)
        xscore_new = np.delete(xscore_new,remove,axis=1)
        
        val_pred = mod.predict(np.array(xscore_new))
        val_prob = pd.DataFrame(mod.predict_proba(np.array(xscore_new)))
        val_prob = val_prob.add_prefix(scoring_prob_name)
        
        output = xscore_dep[['dep_servicevisit']]
        output[scoring_name] = val_pred
        output = pd.concat([output,val_prob], axis = 1)
        print(classification_report(output['dep_servicevisit'], output[scoring_name]))
        output = pd.concat([raw2,output],axis = 1)
        return output
        
#%% Model Regression-------------------------------SPEND-------------------------------------------
    
def BuildSpendRF(xtrn, ytrn, xtst, ytst):
    clf=RandomForestClassifier(n_jobs = -1)


    clf.fit(xtrn,ytrn)
    y_train_pred = clf.predict(xtrn)
    y_pred=clf.predict(xtst)
    y_train_pred = clf.predict(xtrn)
    
    print("Train Accuracy is :",metrics.accuracy_score(ytrn, y_train_pred))
    print("Test Accuracy is :",metrics.accuracy_score(ytst, y_pred))
    
    print("Train ")
    print(classification_report(ytrn, y_train_pred)) 
    
    print("\nTest ")
    print(classification_report(ytst, y_pred)) 
    return clf    
    


def ScoreSpendRFModel(xscore ,mod , traindata, dept, scoring_name, scoring_prob_name):
    
    col = traindata.drop([dept],axis=1).columns.tolist()
    xscore_new = xscore[col]
    xscore_new = np.array(xscore_new)
    
    val_pred = mod.predict(np.array(xscore_new))
    val_prob = pd.DataFrame(mod.predict_proba(np.array(xscore_new)))
    val_prob = val_prob.add_prefix(scoring_prob_name)
    output = pd.DataFrame()
    # output = xscore[['spend_group']]
    output[scoring_name] = val_pred
    output = pd.concat([output,val_prob], axis = 1) 
    # print(classification_report(output['spend_group'], output[scoring_name]))
    
    return output 
       
    
     
#%%

def visitDecileScore(df, prob_filter, name_decile):
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

#%%
    
def bucket_list(df):
    if df['predict_spend'] == 1:
        return 50
    elif df['predict_spend'] == 2:
        return 150
    elif df['predict_spend'] == 3:
        return 250
    elif df['predict_spend'] == 4:
        return 350
    elif df['predict_spend'] == 5:
        return 450
    elif df['predict_spend'] == 6:
       return  750
    elif df['predict_spend'] == 7:
        return 1000

    
