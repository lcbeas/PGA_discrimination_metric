data {
  int<lower=0> N;
  int<lower=0> nplayers;
  int<lower=0> nholes;
  vector[N] y;
  int<lower=1, upper=nplayers> playerid[N];
  int<lower=1, upper=nholes> holeid[N];
}
parameters {
    vector[nplayers] xU;
    vector[nholes] alphaU;
    vector[nholes] betaU;
    vector[nholes] sigma;
}
transformed parameters {
    vector[N] y_hat;
    vector[nplayers] x;
    vector[nholes] alpha;
    vector[nholes] beta;
    vector[N] sigmavec;
    real xmean;
    real xsd;
    
    xmean = mean(xU);
    xsd = sd(xU);
    
    for (i in 1:N){
        y_hat[i] = alphaU[holeid[i]] + betaU[holeid[i]]*xU[playerid[i]];
	sigmavec[i] = sigma[holeid[i]];
	}
	
    for(i in 1:nplayers){
	x[i] = (xU[i] - xmean) / xsd;
   }

    for (j in 1:nholes){
	alpha[j] = alphaU[j] + xmean*betaU[j];
        beta[j] = betaU[j]*xsd;
	}
}

model {
    y ~ normal(y_hat, sigmavec);
    xU ~ normal(0, 1);
    alphaU ~ normal(0, 10);
    betaU ~ normal(0, 10);
    sigma ~ uniform(0,10);
}
