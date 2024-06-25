import pandas as pd
import numpy as np
import torch

def makedata(X):
    for i in range(X.shape[0]):
        if(X[i,1] == 0):
            X[i,0] = 0
            for j in range(7):
                X[i,j+1] = 0
    return X

def prcs_audio(covarep):
    covarep = covarep.iloc[:,:].values
    covarep = makedata(covarep)
    covarep = covarep[covarep.shape[0]-1000:]
    return np.array(covarep)

def return_tensor(covarep):
    return torch.tensor(prcs_audio(covarep)).to(torch.float32)