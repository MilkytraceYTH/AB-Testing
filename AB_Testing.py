#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np


# In[2]:


# read in the data
df = pd.read_csv("AB_test_data.csv")


# In[4]:


# inspect data
df.head()


# In[5]:


df['purchase_TF'].value_counts()


# In[6]:


df['Variant'].value_counts()


# **Hypothesis setup:**
# 
# Null Hypothesis $H_{0}$: Variant B and Variant A had the same conversion rates 
# 
# Alternative hypothesis $H_{A}$: Variant B had a higher conversion rate than Variant A
# 
# **Assumptions:**
# 1. Variant A represents the population and we can treat the population mean as known and equal to the mean of Variant A.

# ##### Conducting the test

# In[4]:


# calculate our z score
p_treatment = df[df["Variant"]=="B"]['purchase_TF'].sum()/len(df[df["Variant"]=="B"]['purchase_TF'])
p_varA = df[df["Variant"]=="A"]['purchase_TF'].sum()/len(df[df["Variant"]=="A"]['purchase_TF'])
n = len(df[df["Variant"]=="B"]['purchase_TF'])
z = (p_treatment-p_varA)/(((p_varA*(1-p_varA))/n)**0.5)
if z > 1.64:
    print("We reject the null. The conversion rate of variant B is significantly higher than that of variant A.")
    print("Our Z score is {}.".format(z))
else:
    print("Test failed. The old version is not that different from the new in terms of conversion rate.")


# With 95% confidence level, $Z_{0.05}$ = 1.64. Reject null if z > 1.64. Since z is 8.7, **we reject the null hypothesis and conclude that at 95% confidence level, variant B generates more conversion than variant A.**

# ##### Optimal Sample Size

# In[5]:


# Calculate optumal sample size
t_alpha = 1.96
t_beta = 0.842
p0 = p_varA
p1 = p_treatment
delta = p1-p0
p_bar = (p0+p1)/2

# plug into the formula
n_star = ((t_alpha*((2*p_bar*(1-p_bar))**.5)+(t_beta*((p0*(1-p0)+p1*(1-p1))**.5)))**2)*(delta**-2)
print("The optimal sample size for each segment is {}".format(n_star))


# In[6]:


# seperate treatment and control groups
A = df[df["Variant"]=="A"]
B = df[df["Variant"]=="B"]


# In[29]:


# test using 1-sample 

log = []
sample_list = []
for i in range(10):
    n = 1158

    sample_B = B.sample(n=n)
    sample_list.append(sample_B)


    convB = sample_B['purchase_TF'].sum()/n

    z_sample = (convB-p_varA)/(((p_varA*(1-p_varA))/n)**0.5)
    if z_sample >= 1.64:
        log.append(1) # reject Null - Variant B is better 
   
    else:
        log.append(0) # fail to reject Null - Variant B is NOT better 
        
        
print("The challenger wins {}% of the time.".format(sum(log)/len(log)*100))


# ##### Sequential Testing

# Assume P(Xi=1) under H0 = p-varA and P(Xi=1) under H1 = p-treatment.
# 
# Set desired type 1 error = 5% and type 2 error = 20%.

# In[30]:


# using the same sample as in part 2

# set parameters 
n = 1158
n_trials = 10
alpha = .05
beta = .2
min_diff = p_treatment-p_varA # from original dataset
upper_bound = np.log(1/alpha)
lower_bound = np.log(beta)

# test each observation in the sample:
list_of_trials = []
number_of_success = 0

for j in range(len(sample_list)):
    sample_B = sample_list[j]
    log_lambda_n = 0
    for i in range(len(sample_B)):  
        
        # update log_lambda_n
        if sample_B['purchase_TF'].iloc[i] == True:
            log_lambda_xi = np.log(p_treatment/p_varA)               
        else:
            log_lambda_xi = np.log((1-p_treatment)/(1-p_varA))

        log_lambda_n += log_lambda_xi
        
        
        # check if log_lambda_n is out of bounds
        if log_lambda_n > upper_bound: # Accept H1
            number_of_success += 1
            break 
        
        elif log_lambda_n < lower_bound: # Accept H0
            break
            
    list_of_trials.append(i+1) # record the number of iterations required to stop test 
    
print("Success rate is {}%".format(number_of_success/n_trials*100))


# In[31]:


# get the avg number of iterations required to stop the test
np.mean(list_of_trials)


# In[33]:


# examine the list of trial number 
list_of_trials

