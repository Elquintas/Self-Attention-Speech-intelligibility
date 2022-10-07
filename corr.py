import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

def rmse(a,b):
    return np.sqrt(((a-b)**2).mean())

def plot_graph(a, b, corr_measure,rmse):
    
    a = np.asarray(a)*10
    b = np.asarray(b)*10

    vec = 10*np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,  0.9,1.0])
    fig,ax = plt.subplots()
    plt.plot(a,b,'o',color='blue',markersize=15)

    plt.plot([0,10],[0,10],zorder=1,color='green', linestyle='dashed', label='y=x',linewidth=1.5)
    
    plt.xticks(np.arange(0.0,10.5,1),fontsize=28)
    plt.yticks(np.arange(0.0,10.5,1),fontsize=28)

    plt.gca().set_aspect(0.75)

    mm, bb = np.polyfit(a,b,1)
    plt.plot(vec, mm*vec + bb, linestyle='dashdot', color='blue', label='regression line',linewidth=1.5)
    plt.xlabel('Intelligibility - Reference',fontsize=36)
    plt.ylabel('Intelligibility - Predicted', fontsize=36)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    plt.plot([], [], ' ', label="p = {:.2f}".format(corr_measure))
    plt.plot([], [], ' ', label="RMSE = {:.3f}".format(rmse*10))
    plt.legend(fontsize=26)
    plt.show()

def compute_corr(file_i,n):
    
    data = np.genfromtxt(file_i,delimiter=',')
    filenames = pd.read_csv(file_i,sep=',',header=None)
    filenames = filenames[0].values.tolist() 
    
    iterator = n #n
    start = 0
    predicted_mean = []
    predicted_median = []
    predicted_std = []
    predicted_mode = []
    reference = [] 
    names = []

    while(start < len(data)):

        buf_mean = np.sum(data[start:start+iterator,2])/iterator    #[start:start+iterator]) #/iterator
        buf_median = np.median(data[start:start+iterator,2])
        buf_std = np.std(data[start:start+iterator,2])
        buf_mode = stats.mode(data[start:start+iterator,2])
        
        predicted_mean.append(round(buf_mean,4))
        predicted_median.append(round(buf_median,4))
        predicted_std.append(round(buf_std,4))
        predicted_mode.append(buf_mode[0])

        names.append(filenames[start])
        
        #reference.append(data[start,1])
        
        ref_mean = np.sum(data[start:start+iterator,1])/iterator
        reference.append(round(ref_mean,4))
        start = start+iterator


    corr_mean = stats.spearmanr(reference,predicted_mean)
    rmse_mean = rmse(np.asarray(reference),np.asarray(predicted_mean))
    corr_median = stats.spearmanr(reference,predicted_median)
    rmse_median = rmse(np.asarray(reference),np.asarray(predicted_median))
    corr_mode = stats.spearmanr(reference,predicted_mode)
    rmse_mode = rmse(np.asarray(reference),np.asarray(predicted_mode))

    #ccc = (2*corr_pearson[0]*std_ref*std_pred)/(std_ref**2 + std_pred**2 + (mean_ref-mean_pred)**2)

    MAE = np.round(np.abs(np.asarray(predicted_mean)-np.asarray(reference)),4)

    with open('results.csv', 'w') as fp:
        for (name,ref,mean,std,mae) in zip(names,reference,predicted_mean,predicted_std,MAE):
            fp.writelines("{0},{1},{2},{3},{4}\n".format(name,ref,mean,std,mae))
    fp.close()    

    #print('Concordance Correlation Coefficient: {}'.format(ccc))
    print('Spearman\'s Correlation Coefficient: {}'.format(stats.spearmanr(reference,predicted_mean)))
    print('Root Mean Squared Error: {}'.format(rmse_mean))
    
    #plot_graph(reference,predicted_mean,corr_mean[0],rmse_mean)
    return reference, predicted_mean

