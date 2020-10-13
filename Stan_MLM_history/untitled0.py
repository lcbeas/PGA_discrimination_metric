# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:59 2019

@author: Luke
"""

import pystan
import matplotlib
import scipy
import pandas as pd
from sklearn import preprocessing, datasets, linear_model, metrics
import numpy as np  



df = pd.DataFrame({"Player": [1,2,2,3,1,2,3,4],'Course': [1,1,2,2,3,3,3,3], 'Score':  [75,74,72,71,68,68,71,70]})

le = preprocessing.LabelEncoder()
df['Player'] = le.fit_transform(df['Player'])
df['Course'] = le.fit_transform(df['Course'])



ml_code = """
data {
    int<lower=0> N; 
  int<lower=1,upper=4> players[N];
  int<lower=1,upper=3> course[N];
  vector[N] dist;
}
transformed data {}
parameters {
    vector<lower=0,upper=7>[4] p;
    vector<lower=-7,upper=0>[3] c;
    real<lower=0> constant;
    real<lower=0> sigma;
}
transformed parameters {
     }   
model {
       sigma ~ normal(0,1);
    
    dist ~ normal(constant + c[course] + p[players], sigma);
}
"""

ml_data = {    'N': len(df['Score']),
             'players': df['Player']+1,
             'course': df['Course']+1,
             'dist': df['Score']
            }

fit = pystan.stan(model_code=ml_code, data=ml_data, iter=1000, chains=1)

print("test")
la = fit.extract(permuted=True)  # return a dictionary of arrays

## return an array of three dimensions: iterations, chains, parameters
a = fit.extract(permuted=False)
print(fit)
fit.plot()