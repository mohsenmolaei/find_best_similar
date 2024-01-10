import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import csv
import pdb

def KDE(data):
    data = data.reshape(-1, 1)
    # Fit kernel density estimator
    kde = KernelDensity(bandwidth=0.75, kernel='gaussian')
    kde.fit(data)
        # Generate values for the x-axis
    x_values = np.linspace(-0.05, 0.05, 22).reshape(-1, 1)

    # Compute the log-likelihoods for each point
    log_densities = kde.score_samples(x_values)

    return np.exp(log_densities)


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def read_data():

# Open the CSV file
    dataArray = []
    with open('top15_crypto.csv', 'r') as file:
    # Create a CSV reader object
        reader = csv.reader(file)
        headers = next(reader)
        
    # Find the index of the desired column
        desired_column = 'BTC-USD' #'change'
        column_index = headers.index(desired_column)
        for row in reader:
                dataArray.append(float(row[column_index])*100) 
        return dataArray

random_array = np.array(read_data())

random_array =np.array( pd.DataFrame(random_array).pct_change())
random_array[0] = 0
# random_array =  (random_array[1:]-random_array[0:-1])/random_array[0:-1]


def fbs(ref_data,test_data,seq_num):
    argmin_kl = float('inf')
    argmin_point = 0
    kl=0
    for i in range(0,len(ref_data)-seq_num+1,seq_num):
        
        for sub_count in range(0,seq_num-9,1):
            kl += kl_divergence(KDE(test_data[sub_count:sub_count+10]),KDE(ref_data[i+sub_count:i+sub_count+10]))# P to q OR q to p kl ???????
        if kl < argmin_kl:
            argmin_kl = kl
            argmin_point = i
           
    return argmin_kl,argmin_point
# my_array = np.arange(1, len(random_array)+1)

ref_data = random_array[:-60]
test_data = random_array[-60:]

argmin_kl,argmin_point = fbs(ref_data,test_data,60)


plt.plot(ref_data[argmin_point:argmin_point+60], label='plot')
plt.plot(test_data, label='plot')

# plt.axvline(x=argmin_point, color='blue', linestyle='--')
# plt.axvline(x=argmin_point + 20, color='blue', linestyle='--')
# plt.axvline(x=len(random_array)- 1, color='red', linestyle='--')
# plt.axvline(x=len(random_array)- 20, color='red', linestyle='--')
# plt.show()


