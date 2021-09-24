# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:04:25 2020

Functions to Run all 4 models [retail/Lease inmarket, retail/Lease Defection]
"""

import PPOFunctions
import pandas as pd
import numpy as np
import statsmodels.api as sm

#%%

def RunRetailInmarket():
    retail_in_t, retail_in_s = PPOFunctions.retail_inmarket_data()
    retail_in_t = PPOFunctions.reduce_mem_usage(retail_in_t)
    retail_in_s = PPOFunctions.reduce_mem_usage(retail_in_s)
    train = PPOFunctions.data_prep_retail_inmarket(retail_in_t)
    score = PPOFunctions.data_prep_retail_inmarket(retail_in_s)
    scale_train = PPOFunctions.scaling_retail_inmarket(train)
    scale_score = PPOFunctions.scaling_retail_inmarket(score)
    Xtrain, ytrain, Xtest, ytest = PPOFunctions.SamplingAndSplit(df = scale_train, dept='retail_is_in_market')
    reject_feature_position,reject_feature, pvals = PPOFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=scale_train, dept = 'retail_is_in_market')
    model = PPOFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 
    output = PPOFunctions.ScorelogModel(xscore = scale_score ,mod = model ,remove = reject_feature_position, 
                                    traindata = scale_train, dept='retail_is_in_market', 
                                    scoring_name='prediction_inmarket', scoring_prob_name='in_market_prob_')
    decile, decile_summary = PPOFunctions.InmarketDecileScore(df = output, 
                                                  prob_filter = 'in_market_prob_1', 
                                                  name_decile = 'in_market_decile')
    return decile, decile_summary, pvals
    

def RunLeaseInmarket():
    lease_in_t, lease_in_s = PPOFunctions.lease_inmarket_data()
    lease_in_t = PPOFunctions.reduce_mem_usage(lease_in_t)
    lease_in_s = PPOFunctions.reduce_mem_usage(lease_in_s)
    train = PPOFunctions.data_prep_lease_inmarket(lease_in_t)
    score = PPOFunctions.data_prep_lease_inmarket(lease_in_s)
    outlier_train = PPOFunctions.outlier_lease_inmarket(train)
    outlier_score = PPOFunctions.outlier_lease_inmarket(score)
    scale_train = PPOFunctions.scaling_lease_inmarket(outlier_train)
    scale_score = PPOFunctions.scaling_lease_inmarket(outlier_score)
    Xtrain, ytrain, Xtest, ytest = PPOFunctions.SamplingAndSplit(df = scale_train, dept='lease_is_in_market')
    reject_feature_position,reject_feature, pvals = PPOFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=scale_train, dept = 'lease_is_in_market')
    model = PPOFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 
    output = PPOFunctions.ScorelogModel(xscore = scale_score ,mod = model ,remove = reject_feature_position, 
                                    traindata = scale_train, dept='lease_is_in_market', 
                                    scoring_name='prediction_inmarket', scoring_prob_name='in_market_prob_')
    decile, decile_summary = PPOFunctions.InmarketDecileScore(df = output, 
                                                  prob_filter = 'in_market_prob_1', 
                                                  name_decile = 'in_market_decile')
    return decile, decile_summary, pvals


def RunRetailDefection():
    retail_de_t, retail_de_s = PPOFunctions.retail_defection_data()
    retail_de_t = PPOFunctions.reduce_mem_usage(retail_de_t)
    retail_de_s = PPOFunctions.reduce_mem_usage(retail_de_s)
    train = PPOFunctions.data_prep_retail_defection(retail_de_t)
    score = PPOFunctions.data_prep_retail_defection(retail_de_s)
    outlier_train = PPOFunctions.outlier_retail_defection(train)
    outlier_score = PPOFunctions.outlier_retail_defection(score)
    scale_train = PPOFunctions.scaling_retail_defection(outlier_train)
    scale_score = PPOFunctions.scaling_retail_defection(outlier_score)
    Xtrain, ytrain, Xtest, ytest = PPOFunctions.SamplingAndSplit(df = scale_train, dept='did_churn_retail')
    reject_feature_position,reject_feature, pvals = PPOFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=scale_train, dept = 'did_churn_retail')
    model = PPOFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 
    output = PPOFunctions.ScorelogModel(xscore = scale_score ,mod = model ,remove = reject_feature_position, 
                                    traindata = scale_train, dept='did_churn_retail', 
                                    scoring_name='prediction_defection', scoring_prob_name='defection_prob_')
    decile, decile_summary = PPOFunctions.DefectionDecileScore(df = output, 
                                                  prob_filter = 'defection_prob_1', 
                                                  name_decile = 'defection_decile')
    return decile, decile_summary, pvals

    
def RunLeaseDefection():
    lease_de_t, lease_de_s = PPOFunctions.lease_defection_data()
    lease_de_t = PPOFunctions.reduce_mem_usage(lease_de_t)
    lease_de_s = PPOFunctions.reduce_mem_usage(lease_de_s)
    train = PPOFunctions.data_prep_lease_defection(lease_de_t)
    score = PPOFunctions.data_prep_lease_defection(lease_de_s)
    outlier_train = PPOFunctions.outlier_lease_defection(train)
    outlier_score = PPOFunctions.outlier_lease_defection(score)
    scale_train = PPOFunctions.scaling_lease_defection(outlier_train)
    scale_score = PPOFunctions.scaling_lease_defection(outlier_score)
    Xtrain, ytrain, Xtest, ytest = PPOFunctions.SamplingAndSplit(df = scale_train, dept='did_churn_lease')
    reject_feature_position,reject_feature, pvals = PPOFunctions.RunLogit(ytrain=ytrain,xtrain = Xtrain, scaleTrain=scale_train, dept = 'did_churn_lease')
    model = PPOFunctions.BuildlogModel(Xtrain, ytrain, Xtest, ytest,reject_feature_position ) 
    output = PPOFunctions.ScorelogModel(xscore = scale_score ,mod = model ,remove = reject_feature_position, 
                                    traindata = scale_train, dept='did_churn_lease', 
                                    scoring_name='prediction_defection', scoring_prob_name='defection_prob_')
    decile, decile_summary = PPOFunctions.DefectionDecileScore(df = output, 
                                                  prob_filter = 'defection_prob_1', 
                                                  name_decile = 'defection_decile')
    return decile, decile_summary, pvals
#%%
