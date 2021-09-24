import SOCIFunctions
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pdb
import gc
# pdb.set_trace()
#%% Load the data
def bnso_retail():
    print("bnso_retail----STARTED")
    # retail data load
    retail_bnso_t, retail_bnso_s = SOCIFunctions.BNSOretail()
    # reducing memory
    retail_bnso_t = SOCIFunctions.reduce_mem_usage(retail_bnso_t)
    retail_bnso_s = SOCIFunctions.reduce_mem_usage(retail_bnso_s)
    # pdb.set_trace()
    # Data backup
    raw2 = retail_bnso_s.copy()
    
    ## Data Processing
    retail_bnso_t = SOCIFunctions.BNSO_retail_dataprep(df = retail_bnso_t)
    retail_bnso_s = SOCIFunctions.BNSO_retail_dataprep(df = retail_bnso_s)
    # pdb.set_trace()
    ## Outlier
    retail_bnso_t, col_1 = SOCIFunctions.outlier_generic(retail_bnso_t,dept = 'dep_servicevisit', ty='tr' )
    retail_bnso_s, col_2 = SOCIFunctions.outlier_generic(retail_bnso_s,dept = 'dep_servicevisit' )
    
    retail_bnso_t, retail_bnso_s = SOCIFunctions.scaling_generic(retail_bnso_t,retail_bnso_s, col_1 )
    
    ## Feature Selection
    retail_bnso_t,col_to_remove = SOCIFunctions.DataCleaner(retail_bnso_t)
    retail_bnso_s = retail_bnso_s.drop(col_to_remove, axis=1)
    retail_bnso_t = SOCIFunctions.uncorrelated_features(retail_bnso_t, dept = 'dep_servicevisit' , threshold= 0.7)
    
    retail_bnso_t, retail_bnso_s = SOCIFunctions.DataCommon(retail_bnso_t, retail_bnso_s)
    print(retail_bnso_t.shape, retail_bnso_s.shape)
    
    
    ## Modelling
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.SamplingAndSplit(df = retail_bnso_t, dept='dep_servicevisit')
    reject_feature_position,reject_feature, pvals = SOCIFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=retail_bnso_t, dept = 'dep_servicevisit')
    model = SOCIFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position )
    
    #Scoring
    output = SOCIFunctions.ScorelogModel(xscore = retail_bnso_s ,mod = model ,remove = reject_feature_position, 
                                traindata = retail_bnso_t, dept='dep_servicevisit', 
                                scoring_name='prediction_visit', scoring_prob_name='visit_prob_')
   
    
    columns_to_add = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag','purchase_type','mailable','dealer_assigned']
    
    output = pd.concat([raw2[columns_to_add],output],axis = 1)
    

    del retail_bnso_t,retail_bnso_s,Xtrain, ytrain, Xtest, ytest,reject_feature_position,reject_feature,model
    print("Bnso_retail------COMPLETED")
    return output,pvals
#%%  

##Retail BNSO
# 'fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag','purchase_type',
def bnso_lease():
    print("bnso_lease-----STARTED")
    lease_bnso_t, lease_bnso_s = SOCIFunctions.BNSOlease()
    lease_bnso_t = SOCIFunctions.reduce_mem_usage(lease_bnso_t)
    lease_bnso_s = SOCIFunctions.reduce_mem_usage(lease_bnso_s)
    # pdb.set_trace()
    col_to_keep = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag','purchase_type','mailable','dealer_assigned']
    lease_bnso_t = SOCIFunctions.BNSO_lease_dataprep(df = lease_bnso_t)
    lease_bnso_s = SOCIFunctions.BNSO_lease_dataprep(df = lease_bnso_s)
    
    lease_bnso_t = lease_bnso_t.drop(col_to_keep,axis=1)
    lease_bnso_t.clip(lower=0,inplace = True)
    raw2 = lease_bnso_s[col_to_keep].copy()
    lease_bnso_s = lease_bnso_s.drop(col_to_keep,axis=1)
    lease_bnso_s.clip(lower=0,inplace = True)

    lease_bnso_t, col_1 = SOCIFunctions.outlier_generic(lease_bnso_t,dept = 'dep_servicevisit', ty='tr' )
    lease_bnso_s, col_2 = SOCIFunctions.outlier_generic(lease_bnso_s,dept = 'dep_servicevisit' )
    
    lease_bnso_t, lease_bnso_s = SOCIFunctions.scaling_generic(lease_bnso_t,lease_bnso_s, col_1 )
    
    lease_bnso_t,col_to_remove = SOCIFunctions.DataCleaner(lease_bnso_t)
    lease_bnso_s = lease_bnso_s.drop(col_to_remove, axis=1)
    
    lease_bnso_t = SOCIFunctions.uncorrelated_features(lease_bnso_t, dept = 'dep_servicevisit' , threshold= 0.7)

    lease_bnso_t, lease_bnso_s = SOCIFunctions.DataCommon(lease_bnso_t, lease_bnso_s)
    print(lease_bnso_t.shape, lease_bnso_s.shape)
    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.SamplingAndSplit(df = lease_bnso_t, dept='dep_servicevisit')
    reject_feature_position,reject_feature, pvals = SOCIFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=lease_bnso_t, dept = 'dep_servicevisit')
    
    model = SOCIFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position,typ="LEASE") 

    output = SOCIFunctions.ScorelogModel(xscore = lease_bnso_s ,mod = model ,remove = reject_feature_position, 
                                    traindata = lease_bnso_t, dept='dep_servicevisit', 
                                    scoring_name='prediction_visit', scoring_prob_name='visit_prob_')


    output = pd.concat([raw2,output],axis = 1)
    
    del lease_bnso_t,lease_bnso_s,raw2,Xtrain, ytrain, Xtest, ytest,reject_feature_position,reject_feature,model
    
    print("BNSO_lease----COmpleted")
    return output,pvals
 
    
#%% BUSO without service
def buso_wo_service():
    print("bnso_lease wo_service-----STARTED")
    buso_wo_t, buso_wo_s = SOCIFunctions.BUSO_withoutservice()
    
    buso_wo_t = SOCIFunctions.reduce_mem_usage(buso_wo_t)
    buso_wo_s = SOCIFunctions.reduce_mem_usage(buso_wo_s)
    # pdb.set_trace()
    raw2 = buso_wo_s.copy()
    
    
    buso_wo_t = SOCIFunctions.BUSO_service_dataprep(df = buso_wo_t) 
    buso_wo_s = SOCIFunctions.BUSO_service_dataprep(df = buso_wo_s)
    
    buso_wo_t, col_1 = SOCIFunctions.outlier_generic(buso_wo_t,dept = 'dep_servicevisit', ty='tr' )
    buso_wo_s, col_2 = SOCIFunctions.outlier_generic(buso_wo_s,dept = 'dep_servicevisit' )
    
    buso_wo_t, buso_wo_s = SOCIFunctions.scaling_generic(buso_wo_t,buso_wo_s,col_1 )
    
    buso_wo_t,col_to_remove = SOCIFunctions.DataCleaner(buso_wo_t)
    buso_wo_s = buso_wo_s.drop(col_to_remove, axis=1)
    
    buso_wo_t = SOCIFunctions.uncorrelated_features(buso_wo_t, dept = 'dep_servicevisit' , threshold= 0.7)

    buso_wo_t, buso_wo_s = SOCIFunctions.DataCommon(buso_wo_t, buso_wo_s)
    print(buso_wo_t.shape, buso_wo_s.shape)

    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.SamplingAndSplit(df = buso_wo_t, dept='dep_servicevisit')
    reject_feature_position,reject_feature, pvals = SOCIFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=buso_wo_t, dept = 'dep_servicevisit')
    model = SOCIFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 

    columns_to_add = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag','purchase_type','mailable','dealer_assigned']
    
    buso_wo_s = pd.concat([raw2[columns_to_add],buso_wo_s],axis = 1)
    # pdb.set_trace()
    output_1 = SOCIFunctions.ScorelogModel(xscore = buso_wo_s ,mod = model ,remove = reject_feature_position, 
                                    traindata = buso_wo_t, dept='dep_servicevisit', 
                                    scoring_name='prediction_visit', scoring_prob_name='visit_prob_',
                                    mod_type_dic={"mod_type":"BUSO","mail_flag":1,"col_keep":columns_to_add})
    output_0 = SOCIFunctions.ScorelogModel(xscore = buso_wo_s ,mod = model ,remove = reject_feature_position, 
                                    traindata = buso_wo_t, dept='dep_servicevisit', 
                                    scoring_name='prediction_visit', scoring_prob_name='visit_prob_',
                                    mod_type_dic={"mod_type":"BUSO","mail_flag":0,"col_keep":columns_to_add})
    
    output = pd.concat([output_1,output_0],axis=0)
    del buso_wo_t, buso_wo_s, Xtrain, ytrain, Xtest, ytest, raw2, model, output_1, output_0,reject_feature_position,reject_feature
    print("buso_lease wo_service-----COMPLETED")
    return output,pvals

#%% BUSO with service
def buso_ws_service():
    print("buso_ws_service-----STARTED")
    buso_ws_t, buso_ws_s = SOCIFunctions.BUSO_withservice()
    # pdb.set_trace()
    buso_ws_t = SOCIFunctions.reduce_mem_usage(buso_ws_t)
    buso_ws_s = SOCIFunctions.reduce_mem_usage(buso_ws_s)
    # pdb.set_trace()
    raw2 = buso_ws_s.copy()
        
    buso_ws_t = SOCIFunctions.BUSO_service_dataprep(df = buso_ws_t) 
    buso_ws_s = SOCIFunctions.BUSO_service_dataprep(df = buso_ws_s)
    
    buso_ws_t, col_1 = SOCIFunctions.outlier_generic(buso_ws_t,dept = 'dep_servicevisit', ty='tr' )
    buso_ws_s, col_2 = SOCIFunctions.outlier_generic(buso_ws_s,dept = 'dep_servicevisit' )
    
    buso_ws_t, buso_ws_s = SOCIFunctions.scaling_generic(buso_ws_t,buso_ws_s,col_1 )
    
    buso_ws_t,col_to_remove = SOCIFunctions.DataCleaner(buso_ws_t)
    buso_ws_s = buso_ws_s.drop(col_to_remove, axis=1)
    
    buso_ws_t = SOCIFunctions.uncorrelated_features(buso_ws_t, dept = 'dep_servicevisit' , threshold= 0.7)
    
    buso_ws_t, buso_ws_s = SOCIFunctions.DataCommon(buso_ws_t, buso_ws_s)
    print(buso_ws_t.shape, buso_ws_s.shape)

    
    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.SamplingAndSplit(df = buso_ws_t, dept='dep_servicevisit')
    reject_feature_position,reject_feature, pvals = SOCIFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=buso_ws_t, dept = 'dep_servicevisit')
    model = SOCIFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 
    
    columns_to_add = ['fca_id','i_consmr','i_hshld_id_curr', 'vin', 'new_used_flag','purchase_type','mailable','dealer_assigned']
    
    buso_ws_s = pd.concat([raw2[columns_to_add],buso_ws_s],axis = 1)   
    output_1 = SOCIFunctions.ScorelogModel(xscore = buso_ws_s ,mod = model ,remove = reject_feature_position, 
                                    traindata = buso_ws_t, dept='dep_servicevisit', 
                                    scoring_name='prediction_visit', scoring_prob_name='visit_prob_',
                                    mod_type_dic={"mod_type":"BUSO","mail_flag":1,"col_keep":columns_to_add})
    
    output_0 = SOCIFunctions.ScorelogModel(xscore = buso_ws_s ,mod = model ,remove = reject_feature_position, 
                                    traindata = buso_ws_t, dept='dep_servicevisit', 
                                    scoring_name='prediction_visit', scoring_prob_name='visit_prob_',
                                    mod_type_dic={"mod_type":"BUSO","mail_flag":0,"col_keep":columns_to_add})
    
    output = pd.concat([output_1,output_0],axis=0)
    # pdb.set_trace()
    del buso_ws_t,buso_ws_s,raw2,Xtrain, ytrain, Xtest, ytest,reject_feature_position,reject_feature,model
    print("buso_ws_service-----COMPLETED")
    return output,pvals

#%% BNSO spends

def RunBNSOSpend():
    print("########### Running BNSO Spend Model################")
    bnso_spend_t, bnso_spend_s = SOCIFunctions.BNSO_spend()
    
    bnso_spend_t = SOCIFunctions.reduce_mem_usage(bnso_spend_t)
    bnso_spend_s = SOCIFunctions.reduce_mem_usage(bnso_spend_s)
    # pdb.set_trace()
    raw2 = bnso_spend_s.copy()
        
    bnso_spend_t = SOCIFunctions.BNSO_spend_dataprep(df = bnso_spend_t, ty = 'tr') 
    bnso_spend_s = SOCIFunctions.BNSO_spend_dataprep(df = bnso_spend_s) 
    
    bnso_spend_t, col_1 = SOCIFunctions.outlier_generic(bnso_spend_t,dept = 'spend_group', ty='tr' )
    bnso_spend_s, col_2 = SOCIFunctions.outlier_generic(bnso_spend_s )
    
    bnso_spend_t, bnso_spend_s = SOCIFunctions.scaling_generic(bnso_spend_t,bnso_spend_s, col_1 )
    
    bnso_spend_t,col_to_remove = SOCIFunctions.DataCleaner(bnso_spend_t)
    bnso_spend_s = bnso_spend_s.drop(col_to_remove, axis=1)
    
    bnso_spend_t = SOCIFunctions.uncorrelated_features(bnso_spend_t, dept = 'spend_group' , threshold= 0.6)
    bnso_spend_s["spend_group"]=0
    bnso_spend_t, bnso_spend_s = SOCIFunctions.DataCommon(bnso_spend_t, bnso_spend_s) # this is dropping spend_group from training
    bnso_spend_s = bnso_spend_s.drop("spend_group",axis=1)
    print(bnso_spend_t.shape, bnso_spend_s.shape)
    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.StratifySampling(df = bnso_spend_t,dept='spend_group')
    model = SOCIFunctions.BuildSpendRF(Xtrain, ytrain, Xtest, ytest ) 
    
    output = SOCIFunctions.ScoreSpendRFModel(xscore = bnso_spend_s ,mod = model , 
                                    traindata = bnso_spend_t, dept='spend_group',
                                    scoring_name='predict_spend', scoring_prob_name='spend_prob_')
    
    output = pd.concat([raw2[['fca_id','vin']],output],axis=1)
    
    feature_imp = pd.Series(model.feature_importances_,index=bnso_spend_t.drop('spend_group',axis=1).columns).sort_values(ascending=False)
    
    del bnso_spend_t,bnso_spend_s
    return output, feature_imp
    
def RunBUSO_ws_Spend():
    print("########### Running BUSO ws Spend Model################")
    
    buso_ws_spend_t, buso_ws_spend_s = SOCIFunctions.BUSO_ws_spend()
    
    buso_ws_spend_t = SOCIFunctions.reduce_mem_usage(buso_ws_spend_t)
    buso_ws_spend_s = SOCIFunctions.reduce_mem_usage(buso_ws_spend_s)
    # pdb.set_trace()
    raw2 = buso_ws_spend_s.copy()
        
    buso_ws_spend_t = SOCIFunctions.BNSO_spend_dataprep(df = buso_ws_spend_t, ty = 'tr') 
    buso_ws_spend_s = SOCIFunctions.BNSO_spend_dataprep(df = buso_ws_spend_s)
    
    buso_ws_spend_t, col_1 = SOCIFunctions.outlier_generic(buso_ws_spend_t,dept = 'spend_group', ty='tr' )
    buso_ws_spend_s, col_2 = SOCIFunctions.outlier_generic(buso_ws_spend_s )
    
    buso_ws_spend_t, buso_ws_spend_s = SOCIFunctions.scaling_generic(buso_ws_spend_t,buso_ws_spend_s, col_1 )
    
    buso_ws_spend_t,col_to_remove = SOCIFunctions.DataCleaner(buso_ws_spend_t)
    buso_ws_spend_s = buso_ws_spend_s.drop(col_to_remove, axis=1)
    
    buso_ws_spend_t = SOCIFunctions.uncorrelated_features(buso_ws_spend_t, dept = 'spend_group' , threshold= 0.6)
    
    buso_ws_spend_s["spend_group"] = 0
    buso_ws_spend_t, buso_ws_spend_s = SOCIFunctions.DataCommon(buso_ws_spend_t, buso_ws_spend_s)
    buso_ws_spend_s = buso_ws_spend_s.drop("spend_group",axis=1)
    
    print(buso_ws_spend_t.shape, buso_ws_spend_s.shape)
    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.StratifySampling(df = buso_ws_spend_t,dept='spend_group')
    model = SOCIFunctions.BuildSpendRF(Xtrain, ytrain, Xtest, ytest ) 
    output = SOCIFunctions.ScoreSpendRFModel(xscore = buso_ws_spend_s ,mod = model , 
                                    traindata = buso_ws_spend_t, dept='spend_group',
                                    scoring_name='predict_spend', scoring_prob_name='spend_prob_')
    
    output = pd.concat([raw2[['fca_id','vin']],output],axis=1)
    
    feature_imp = pd.Series(model.feature_importances_,index=buso_ws_spend_t.drop('spend_group',axis=1).columns).sort_values(ascending=False)
    del buso_ws_spend_t,buso_ws_spend_s
    return output, feature_imp

def RunBUSO_wo_Spend():
    print("########### Running BUSO wo Spend Model################")

    buso_wo_spend_t, buso_wo_spend_s = SOCIFunctions.BUSO_wo_spend()
    
    buso_wo_spend_t = SOCIFunctions.reduce_mem_usage(buso_wo_spend_t)
    buso_wo_spend_s = SOCIFunctions.reduce_mem_usage(buso_wo_spend_s)
    pdb.set_trace()
    raw2 = buso_wo_spend_s.copy()
        
    buso_wo_spend_t = SOCIFunctions.BUSO_wo_spend_dataprep(df = buso_wo_spend_t,ty='tr') 
    buso_wo_spend_s = SOCIFunctions.BUSO_wo_spend_dataprep(df = buso_wo_spend_s)
    
    buso_wo_spend_t, col_1 = SOCIFunctions.outlier_generic(buso_wo_spend_t,dept = 'spend_group', ty='tr' )
    buso_wo_spend_s, col_2 = SOCIFunctions.outlier_generic(buso_wo_spend_s)
    
    buso_wo_spend_t, buso_wo_spend_s = SOCIFunctions.scaling_generic(buso_wo_spend_t,buso_wo_spend_s, col_1 )
    
    buso_wo_spend_t,col_to_remove = SOCIFunctions.DataCleaner(buso_wo_spend_t)
    buso_wo_spend_s = buso_wo_spend_s.drop(col_to_remove, axis=1)
    
    buso_wo_spend_t = SOCIFunctions.uncorrelated_features(buso_wo_spend_t, dept = 'spend_group' , threshold= 0.6)
    
    buso_wo_spend_s["spend_group"] = 0
    buso_wo_spend_t, buso_wo_spend_s = SOCIFunctions.DataCommon(buso_wo_spend_t, buso_wo_spend_s)
    buso_wo_spend_s = buso_wo_spend_s.drop("spend_group",axis=1)
    
    print(buso_wo_spend_t.shape, buso_wo_spend_s.shape)
    
    Xtrain, ytrain, Xtest, ytest = SOCIFunctions.StratifySampling(df = buso_wo_spend_t,dept='spend_group')
    model = SOCIFunctions.BuildSpendRF(Xtrain, ytrain, Xtest, ytest ) 
    output = SOCIFunctions.ScoreSpendRFModel(xscore = buso_wo_spend_s ,mod = model , 
                                    traindata = buso_wo_spend_t, dept='spend_group',
                                    scoring_name='predict_spend', scoring_prob_name='spend_prob_')
    
    output = pd.concat([raw2[['fca_id','vin']],output],axis=1)
    
    feature_imp = pd.Series(model.feature_importances_,index=buso_wo_spend_t.drop('spend_group',axis=1).columns).sort_values(ascending=False)
    del buso_wo_spend_t,buso_wo_spend_s
    return output, feature_imp
    

