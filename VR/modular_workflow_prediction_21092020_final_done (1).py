#%%
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
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

print("All Libraries import done")

#%%
#td_api_key = getpass('10729/ddb38a08e082675fb2937419600f5b4f93b0489e')
td_api_key = '10729/ddb38a08e082675fb2937419600f5b4f93b0489e'

os.environ['TD_API_KEY'] = td_api_key
print("Succeeded to set the API key")

os.environ["HTTP_PROXY"] = http_proxy = "http://cag\t8317aa:%s@iproxy.appl.chrysler.com:9090" % parse.quote_plus('Ed12b005@Kal')
os.environ['TD_API_SERVER'] = endpoint = 'https://api.treasuredata.com'

con = td.connect(apikey=td_api_key, endpoint=endpoint, http_proxy_proxy=http_proxy)

database = 'sandbox_data_science'
engine = td.create_engine('presto:{}'.format(database))
print("Connected to {}".format(database))

#%% Import Tables
query_raw_data = ("select * from sandbox_data_science.tmc_product_affn_ads_20200914 where lower(trim(repur_np)) != 'grand caravan' and lower(trim(repur_np)) != 'journey'")
raw_data = td.read_td(query_raw_data,engine)

validation_query = ("select * from sandbox_data_science.tmc_product_affn_20200914_v2")
validation_data = td.read_td(validation_query,engine)
#%% Data quality check
category_input = raw_data[['repur_make','repur_np']].drop_duplicates().sort_values('repur_np')

model_qc  =pd.DataFrame(category_input['repur_np'].value_counts()).reset_index()
#%% filter
df_np = pd.read_csv('C:\\Users\\T8317AA\\Desktop\\ATD\\Affinity\\Final\Validation\\imp_np.csv')
df_np['repur_np'] = df_np['repur_np'].str.lower()
df_np['repur_np'] = df_np['repur_np'].str.strip()
raw_data['repur_np'] = raw_data['repur_np'].str.lower()
raw_data['repur_np'] = raw_data['repur_np'].str.strip()
raw_data = raw_data.merge(df_np, left_on ='repur_np', right_on = 'repur_np', how='inner')
#%% Data Cleaning
raw_data =raw_data[(raw_data['repur_make']!='PLYMOUTH')&(raw_data['repur_make']!='STERLING')].reset_index(drop=True)
raw_data['repur_np'][(raw_data['repur_np']=='2500')|(raw_data['repur_np']=='3500')|(raw_data['repur_np']=='4500')|(raw_data['repur_np']=='5500')] = 'Ram Hd'
raw_data['prev_np'][(raw_data['prev_np']=='2500')|(raw_data['prev_np']=='3500')|(raw_data['prev_np']=='4500')|(raw_data['prev_np']=='5500')] = 'Ram Hd' 
raw_data['repur_np'][raw_data['repur_np'].str.lower() == '124 spider abarth'] = '124 spider'
raw_data['prev_np'][raw_data['prev_np'].str.lower() == '124 spider abarth'] = '124 spider'
raw_data =raw_data[raw_data['repur_np'].str.lower() != '500c'].reset_index(drop=True)

#validation_data['repur_np'][(validation_data['repur_np']=='2500')|(validation_data['repur_np']=='3500')|(validation_data['repur_np']=='4500')|(validation_data['repur_np']=='5500')] = 'Ram Hd'
#validation_data['repur_np'][validation_data['repur_np'].str.lower() == '124 spider abarth'] = '124 Spider'
validation_data['prev_np'][(validation_data['prev_np']=='2500')|(validation_data['prev_np']=='3500')|(validation_data['prev_np']=='4500')|(validation_data['prev_np']=='5500')] = 'Ram Hd'
validation_data['prev_np'][validation_data['prev_np'].str.lower() == '124 spider abarth'] = '124 Spider'
#validation_data['urb_sci_pred_np'][validation_data['urb_sci_pred_np'].str.lower() == '124 spider abarth'] = '124 Spider'
#validation_data['urb_sci_pred_np'][validation_data['urb_sci_pred_np'].str.upper() == 'RAM 2500 / RAM 3500 / RAM 4500 / RAM 5500'] = 'Ram Hd'
#%% model backup
#r = raw_data.copy(deep=True)
#v = validation_data.copy(deep=True)
#%% Functions
def ads_models(raw_data,validation_data):

    lead_data_m1 = raw_data[raw_data['intrt_np'].isnull()==False]
    #variables_to_keep = ['i_consmr','prev_np','prev_make','prev_trim','intrt_np','tenyr_npl_hist','urb_sci_pred_np','repur_np','repur_make']
    variables_to_keep = ['i_consmr','i_mod_yr','prev_np','prev_make','prev_trim','intrt_np','tenyr_npl_hist']
    valid_data_m1 = validation_data[validation_data['intrt_np'].isnull()==False]
    valid_data_m1 = valid_data_m1[variables_to_keep].reset_index(drop=True)

    cols_to_keep_m1 = ['i_consmr','prev_np','prev_make','repur_make','repur_np','intrt_np']
    lead_data_m1 = lead_data_m1[cols_to_keep_m1].reset_index(drop=True)


    lead_data_m1 = lead_data_m1.apply(lambda x:x.astype(str).str.lower())
    valid_data_m1 = valid_data_m1.apply(lambda x:x.astype(str).str.lower())
    
    lead_data_m2 = raw_data[raw_data['intrt_np'].isnull()==True]
    
    valid_data_m2 = validation_data[validation_data['intrt_np'].isnull()==True]
    valid_data_m2 = valid_data_m2[variables_to_keep].reset_index(drop=True)
    
    cols_to_keep_m2 = ['i_consmr','prev_np','prev_make','repur_make', 'repur_np','tenyr_npl_hist']
    lead_data_m2 = lead_data_m2[cols_to_keep_m2].reset_index(drop=True)
    
    lead_data_m2 = lead_data_m2.apply(lambda x:x.astype(str).str.lower())
    valid_data_m2 = valid_data_m2.apply(lambda x:x.astype(str).str.lower())
    
    return lead_data_m1,valid_data_m1,lead_data_m2,valid_data_m2

def ads_prep(lead_data,model_num=1):
    data = lead_data.copy()

    data['dependent'] = data['repur_np'].factorize()[0]
    category_id_df = data[['repur_make','repur_np', 'dependent']].drop_duplicates().sort_values('dependent')

    #category_to_id = dict(category_id_df.values)
    #id_to_category = dict(category_id_df[['dependent', 'repur_np']].values)

    data.drop(['repur_np'],axis=1,inplace =True)

    model_data = pd.get_dummies(data=data, columns=['prev_np'])
    #model_make = pd.get_dummies(data=data[['prev_make']],columns=['prev_make'])
    if(model_num==1):
        name_np = 'intrt_np'
        name_prefix = 'interest_'
    else:
        name_np = 'tenyr_npl_hist'
        name_prefix = 'tenyr_'
    
    interest_np = data[name_np].str.split(',',expand=True)
    interest_np= pd.get_dummies(interest_np, prefix="", prefix_sep="").astype(np.int8)
    interest_np = interest_np.groupby(level=0, axis=1).sum().reset_index(drop=True)
    interest_np = interest_np.add_prefix(name_prefix)
    #model_data = pd.concat([model_data,model_make, interest_np],axis=1)
    model_data = pd.concat([model_data, interest_np],axis=1)
    model_data.drop([name_np,'i_consmr','prev_make'],axis=1, inplace =True)
    
    return model_data,category_id_df
    
def valid_prep(valid_data,model_num=1):
    data_valid = valid_data.copy()
    model_data_valid = pd.get_dummies(data=data_valid, columns=['prev_np'])
    #model_make_valid = pd.get_dummies(data=data_valid[['prev_make']], columns=['prev_make'])
    
    if(model_num==1):
        name_np = 'intrt_np'
        name_prefix = 'interest_'
        drop_list = ['intrt_np','tenyr_npl_hist','i_consmr','prev_make']
    else:
        name_np = 'tenyr_npl_hist'
        name_prefix = 'tenyr_'
        drop_list = ['tenyr_npl_hist','i_consmr','prev_make']
    
    interest_np = data_valid[name_np].str.split(',',expand=True)
    interest_np= pd.get_dummies(interest_np, prefix="", prefix_sep="").astype(np.int8)
    
    interest_np1 = interest_np.iloc[:interest_np.shape[0]//2,:]
    interest_np2 = interest_np.iloc[interest_np.shape[0]//2:,:]
    interest_np1 = interest_np1.groupby(level=0, axis=1).sum().reset_index(drop=True)
    interest_np2 = interest_np2.groupby(level=0, axis=1).sum().reset_index(drop=True)
    interest_np = pd.concat([interest_np1,interest_np2],ignore_index=True)
    interest_np = interest_np.add_prefix(name_prefix)

    #model_data_valid = pd.concat([model_data_valid,model_make_valid, interest_np],axis=1)
    model_data_valid = pd.concat([model_data_valid, interest_np],axis=1)
    model_data_valid.drop(drop_list,axis=1, inplace =True)
    
    return model_data_valid


def model_build(X_features,dep_var,model_data,model_num=1):

    model_test  =pd.DataFrame(model_data['dependent'].value_counts()).reset_index()
    model_test.rename(columns={'dependent':'values','index':'dependent'},inplace=True)
    model_test = model_test[model_test['values']>1].reset_index(drop=True)
    model_data = model_data.merge(model_test[['dependent']], left_on ='dependent', right_on = 'dependent', how='inner')
    
    X = model_data[X_features]
    y = model_data[[dep_var]]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0, test_size = 0.2, stratify = y)
    #clf=RandomForestClassifier(n_estimators = 1000, n_jobs = -1,min_samples_leaf=100)
    #rf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
    #param_grid = {'min_samples_leaf':[10,50,100,500,1000],'max_features':['auto', 'sqrt', 'log2'],'oob_score':[True,False]}
    #rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 2)
    #rf_random.fit(X_train,y_train)
    #best_param = rf_random.best_params_
    #print(best_param)
    if(model_num==1):
        clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1,oob_score= False, min_samples_leaf = 10, max_features = 'auto' )
    else:
        clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1,oob_score= True, min_samples_leaf = 10, max_features = 'auto' )
    clf.fit(X_train,y_train)
    y_train_pred = clf.predict(X_train)
    y_pred=clf.predict(X_test)
    print("Train Accuracy is :",metrics.accuracy_score(y_train, y_train_pred))
    print("Test Accuracy is :",metrics.accuracy_score(y_test, y_pred))
    return clf



'''def validate_model(model_data_valid,valid_data,common_feature,clf,category_id_df,model_num=1,scoring_date = '20190930'):
    X_valid = model_data_valid[common_feature]
    valid_pred=clf.predict(X_valid)
    
    output = valid_data[['i_consmr','urb_sci_pred_np','repur_np','repur_make','prev_np','prev_make']]
    output['urb_sci_pred_np']=output['urb_sci_pred_np'].str.lower()
    output['repur_np']=output['repur_np'].str.lower()
    output['Model'] = model_num
    output['Scoring_date'] = scoring_date
    
    output['pred_nameplate_model'] = valid_pred
    category_id_df = category_id_df.rename(columns={'repur_np':'pred_nameplate','repur_make':'pred_make'})
    output = output.merge(category_id_df, left_on ='pred_nameplate_model', right_on = 'dependent', how='left')
    
    #output = output.rename(columns={'repur_np':'pred_nameplate','repur_make':'pred_make'})
    output.drop(['pred_nameplate_model','dependent'],axis=1,inplace=True)
    
    #con_new = td.connect()
    #td.to_td(output, '{}.{}'.format('sandbox_data_science','tmc_urb_sci_vald_ads_v1_2012_2019'), con_new, if_exists='append',index=False)
    
    return output'''
    
def validate_model(model_data_valid,valid_data,common_feature,clf,category_id_df,model_num=1,scoring_date = '20200215'):
    X_valid = model_data_valid[common_feature]
    
    #valid_pred=clf.predict(X_valid)
    
    
    
    output = valid_data[['i_consmr','i_mod_yr','prev_make','prev_np','prev_trim','intrt_np','tenyr_npl_hist']]
    output = output.rename(columns={'prev_np':'current_nameplate','prev_make':'current_brand','prev_trim':'current_trim'})
    output['Model'] = model_num
    output['Scoring_date'] = scoring_date
    
    #output['pred_nameplate_model'] = valid_pred
    #category_id_df_v1 = category_id_df.copy(deep=True)
    #category_id_df_v1 = category_id_df_v1.rename(columns={'repur_np':'pred_nameplate','repur_make':'pred_make'})
    #output = output.merge(category_id_df_v1, left_on ='pred_nameplate_model', right_on = 'dependent', how='left')
    
    #output = output.rename(columns={'repur_np':'pred_nameplate','repur_make':'pred_make'})
    #output.drop(['pred_nameplate_model','dependent'],axis=1,inplace=True)
    
    pred_output = np.array(clf.predict_proba(X_valid))
    pred_output = np.round(pred_output,3)
    pred_prob_df = pd.DataFrame(np.sort(pred_output,axis=1)[:,-3:],columns=['third_preferred_score','second_preferred_score','first_preferred_score'])
    pred_np_df = pd.DataFrame(np.array(clf.classes_)[np.argsort(pred_output,axis=1)[:,-3:]],columns=['third_preferred_dep','second_preferred_dep','first_preferred_dep'])
    print("Sort Successful")
    pred_np_df = pred_np_df.merge(category_id_df,left_on ='third_preferred_dep', right_on = 'dependent', how='left')
    pred_np_df = pred_np_df.rename(columns={'repur_np':'third_preferred_nameplate','repur_make':'third_preferred_brand'})
    pred_np_df.drop(['third_preferred_dep','dependent'],axis=1,inplace=True)
    pred_np_df = pred_np_df.merge(category_id_df,left_on ='second_preferred_dep', right_on = 'dependent', how='left')
    pred_np_df = pred_np_df.rename(columns={'repur_np':'second_preferred_nameplate','repur_make':'second_preferred_brand'})
    pred_np_df.drop(['second_preferred_dep','dependent'],axis=1,inplace=True)
    pred_np_df = pred_np_df.merge(category_id_df,left_on ='first_preferred_dep', right_on = 'dependent', how='left')
    pred_np_df = pred_np_df.rename(columns={'repur_np':'first_preferred_nameplate','repur_make':'first_preferred_brand'})
    pred_np_df.drop(['first_preferred_dep','dependent'],axis=1,inplace=True)
    #valid_pred=clf.predict(X_valid)
    #pred_class = pd.DataFrame(clf.classes_,columns=['dependent'])
    #pred_class = pred_class.merge(category_id_df, left_on ='dependent', right_on = 'dependent', how='left')
    #pred_output = pd.DataFrame(clf.predict_proba(X_valid),columns=pred_class['repur_np'].values)
    #pred_output = pred_output.round(3)
    #output = valid_data[['i_consmr','prev_vin_i_mod_yr','prev_np','prev_make','urb_sci_pred_np','repur_np','repur_make']]
    output = pd.concat([output,pred_np_df,pred_prob_df],axis=1)
    
    #output['pred_nameplate_model'] = valid_pred
    #output = output.merge(category_id_df, left_on ='pred_nameplate_model', right_on = 'dependent', how='left')
    
    #output = output.rename(columns={'repur_np':'pred_nameplate','repur_make':'pred_make'})
    #output.drop(['pred_nameplate_model','dependent'],axis=1,inplace=True)
    
    con_new = td.connect()
    td.to_td(output, '{}.{}'.format('sandbox_data_science','tmc_product_affn_20200914_output_v2'), con_new, if_exists='append',index=False)
    del output
    #return output
    


def final_model_build(raw_data,validation_data,scoring_date = '20200810'):   
    # Data Preparation
    #raw_data,validation_data = import_tables()
    model_m1,valid_m1,model_m2,valid_m2 = ads_models(raw_data,validation_data)
    
    # Model 1
    model_data_m1,category_id_df_m1 = ads_prep(model_m1,model_num=1)
    model_data_valid_m1 = valid_prep(valid_m1,model_num=1)
    model_data_m1.drop(['repur_make'],axis=1,inplace =True)
    common_feature_m1 = list(set(model_data_m1) & set(model_data_valid_m1))
    clf_m1 = model_build(common_feature_m1,'dependent',model_data_m1,model_num=1)
    #output_m1 = validate_model(model_data_valid_m1,valid_m1,common_feature_m1,clf_m1,category_id_df_m1,model_num=1,scoring_date=scoring_date)
    validate_model(model_data_valid_m1,valid_m1,common_feature_m1,clf_m1,category_id_df_m1,model_num=1,scoring_date=scoring_date)
    print('Model 1 Successful')
    # Model 
    model_data_m2,category_id_df_m2 = ads_prep(model_m2,model_num=2)
    model_data_valid_m2 = valid_prep(valid_m2,model_num=2)
    model_data_m2.drop(['repur_make'],axis=1,inplace =True)
    common_feature_m2 = list(set(model_data_m2) & set(model_data_valid_m2))
    clf_m2 = model_build(common_feature_m2,'dependent',model_data_m2,model_num=2)
    #output_m2 = validate_model(model_data_valid_m2,valid_m2,common_feature_m2,clf_m2,category_id_df_m2,model_num=2,scoring_date=scoring_date)
    validate_model(model_data_valid_m2,valid_m2,common_feature_m2,clf_m2,category_id_df_m2,model_num=2,scoring_date=scoring_date)
    print('Model 2 Successful')
#%%
#final_output = final_model_build(raw_data,validation_data)
final_model_build(raw_data,validation_data,scoring_date = '20200914')

#print(final_output[final_output['repur_np']==final_output['pred_nameplate']].shape[0])
#%%
'''model_m1,valid_m1,model_m2,valid_m2 = ads_models(raw_data,validation_data)
    
# Model 1
model_data_m1,category_id_df_m1 = ads_prep(model_m1,model_num=1)
model_data_valid_m1 = valid_prep(valid_m1,model_num=1)
model_data_m1.drop(['repur_make'],axis=1,inplace =True)
common_feature_m1 = list(set(model_data_m1) & set(model_data_valid_m1))
clf_m1 = model_build(common_feature_m1,'dependent',model_data_m1,model_num=1)

X_valid = model_data_valid_m1[common_feature_m1]
pred_output = np.array(clf_m1.predict_proba(X_valid))
pred_output = np.round(pred_output,3)

pred_np_df = pd.DataFrame(np.array(clf_m1.classes_)[np.argsort(pred_output,axis=1)[:,-3:]],columns=['third_preferred_dep','second_preferred_dep','first_preferred_dep'])'''
