import pandas as pd
import numpy as np
import torch

def processData(data):
    X = data.iloc[:,:].values
    X = np.delete(X, 0, 1)
    X = np.delete(X, 1, 1)
    for i in range(len(X)):
        if(isinstance(X[i][5],str) or isinstance(X[i][7],str)):
            X[i] = np.zeros((1, X.shape[1]))
    return X

def scale_down(X):
  X_new = []
  size = 2
  for i in range(int(X.shape[0]/size)):
    cur_row = X[i*size]
    for j in range(1,size):
      if(i+j < X.shape[0]):
        cur_row += X[i+j]
    cur_row = cur_row/size
    X_new.append(cur_row)
  X_new = np.array(X_new)
  return X_new

def decrease_size(X):
  size = 1000
  if(X.shape[0] < size):
    dif = size - X.shape[0] 
    temp = np.zeros((dif,X.shape[1]))
    X = np.concatenate((X,temp),axis = 0)
  elif(X.shape[0] > size):
    X = X[:1000, :]
  return X

def prcs_video(au, feat, feat3d, gaze, pose):
    au = processData(au)
    feat = processData(feat)
    feat3d = processData(feat3d)
    gaze = processData(gaze)
    pose = processData(pose)

    vid = np.concatenate((au, feat, feat3d, gaze, pose), 1)
    vid = scale_down(vid)
    vid = decrease_size(vid)

    return vid

def return_tensor(au, feat, feat3d, gaze, pose):
   return torch.tensor(prcs_video(au, feat, feat3d, gaze, pose)).to(torch.float32)