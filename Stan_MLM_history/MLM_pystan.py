# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:59 2019

@author: Luke
"""
import pystan
import pandas as pd
import matplotlib as plt
import scipy

df = pd.read_csv('synthetic_data.csv')
player_names = df.Player.unique()

unpooled_model = """data {
  int<lower=0> nholes; 
  vector[nholes] players;
  vector[nholes] holes;
  vector[nholes] dist;
} 
parameters {
  vector[nholes] p;
  vector[nholes] h;
  real avg;
  real<lower=0,upper=100> sigma;
} 
transformed parameters {
  vector[nholes] d_hat;
  for (i in 1:nholes)
    d_hat[i] <- avg  + p[i] + holes[i];
}
model {
  dist ~ normal(d_hat, sigma);
}"""

unpooled_data = {'nholes': 2000,
                 'players': df['Player'], # Stan counts starting at 1
                 'holes': df['Hole'],
                 'dist': df['Distance']}

sm = pystan.StanModel(model_code=unpooled_model) 
unpooled_fit = sm.sampling(data=unpooled_data, iter=1000, chains=2)

unpooled_estimates = pd.Series(unpooled_fit['a'].mean(0), index=player_names)
unpooled_se = pd.Series(unpooled_fit['a'].std(0), index=player_names)
order = unpooled_estimates.sort_values().index
plt.figure(figsize=(18, 6))
plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    plt.plot([i,i], [m-se, m+se], 'b-') 
    plt.xlim(-1,690); 
plt.ylabel('Price estimate (log scale)');plt.xlabel('Ordered category');plt.title('Variation in category price estimates');