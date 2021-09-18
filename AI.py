import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.externals import joblib



scaler = joblib.load('scaler.save') 
model=load_model('model.h5')



def crop_prediction(n,p,k,temp,humidity,ph,rainfall):
  a=np.array([[n,p,k,temp,humidity,ph,rainfall]])
  a=scaler.transform(a)
  predict_x=model.predict(x_train[:1]) 
  classes_x=np.argmax(predict_x,axis=1)

  input=scaler.transform(a)
  predict_x=model.predict(input) 
  classes_x=np.argmax(predict_x,axis=1)

  c=None

  for i in classes_x:
    c=i

  return mapping[c]
