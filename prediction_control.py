#Useful libraries importation  
import os 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
from pandas import read_csv# pour la lecture et l'importation des donnees depuis le fichier DATA

import keras


from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
path= "C:/Users/yodai/Desktop/S7/Projet scientifique/code_prediction"
os.chdir(path)
#traitment des données d'entrée en la forme voulue 
X_dataframe=read_csv('prediction_donnée.txt', header=None)
Y_dataframe=read_csv('brouillon.txt', header=None)
Test_dataframe = read_csv('DATA.txt', header=None)

X_data=X_dataframe.values
X=X_data[:,0:8]
Y_data=X_dataframe.values
Y=Y_data[:,0:8]

X_train=np.reshape(X,(48,3))
Y_train=np.reshape(Y, (144,1))
Test_data=Test_dataframe.values
X_test=Test_data[:,0:5]
Y_test=Test_data[:,5:10]
X_test=np.reshape(X_test, (170,1))
Y_test=np.reshape(Y_test, (170,1))
X_test=np.delete(X_test, np.s_[144:], axis=0)
X_test=np.reshape(X_test, (48,3))
Y_test=np.delete(Y_test, np.s_[144:], axis=0)
print(X_test.shape)
#creation d'elements 3D
def create_dataset (X,Y, time_steps ):
    Xs, ys = [],[]
    
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        ys.append(Y[i+time_steps])
        Xs.append(v)
        #ys.append(y[i])
        
    return np.array(Xs), np.array(ys)
TIME_STEPS = 10
x_train,y_train =create_dataset (X_train,Y_train, TIME_STEPS)
x_test, y_test = create_dataset(X_test,Y_test, TIME_STEPS)


#mdel du reseau de prediction 
def create_model(units, m):
    global x 
    TIME_STEPS = 30
    
    x_train,y_train =create_dataset (X_train,Y_train, TIME_STEPS)
    model = Sequential()
    # First layer of LSTM
    model.add(m (units = units, return_sequences = True, 
                 input_shape = [x_train.shape[1], x_train.shape[2]]))
    model.add(Dropout(0.2)) 
    # Second layer of LSTM
    model.add(m (units = 54))                
    model.add(Dropout(0.2))
    model.add(Dense(units = 1)) 
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model

def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)

    # shuffle = False because the order of the data matters
    history = model.fit(x_train, y_train, epochs = 100, validation_split = 0.2,
                    batch_size = 32, shuffle = False, callbacks = [early_stop])
    return history

def Tracer_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + str(GRU))
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

    def prediction(model):
    prediction = model.predict(x_test)
    prediction= np.delete(prediction, np.s_[32:], axis=0)
    return prediction
#traiment des elements d'entreed du reseau de controle 
Controle_dataframe = read_csv('brouillon.txt', header=None)#recuperation des données du fichier DATA
controle_data = Controle_dataframe.values#transformation du fichier panda en une matrice numpy 

x=resultat_prediction
Controle_X_train=controle_data[:,1] 
Controle_Y_train=controle_data[:,2]
Controle_X_train=np.delete(Controle_X_train,np.s_[171:],axis=0) 
Controle_X_train=np.reshape(Controle_X_train, (57,3)) 
Controle_Y_train=np.reshape(Controle_Y_train, (172,1)) 

Contr_X_test=np.append(x,[0.5])
Contr_Y_test=controle_data[:,4]
#print(Contr_X_test.shape)
Contr_Y_test=np.delete(Contr_Y_test,np.s_[4:],axis=0)
Contr_Y_test=np.reshape(Contr_Y_test, (4,1))
Contr_X_test=np.reshape(Contr_X_test, (11,3))
print(Controle_X_train.shape)
print(Controle_Y_train.shape)
x,y=create_dataset (Controle_X_train,Controle_Y_train, 10)
xtest,ytest=create_dataset(Contr_X_test,Controle_Y_train, 10)
#creation du model de controle 
def create_modelControl(units):
    
    TIME_STEPS = 10
    
    x,y=create_dataset (Controle_X_train,Controle_Y_train, 10)
    model = Sequential()
    # First layer of LSTM
    model.add(Dense (units = units, activation='relu',
                 input_shape = [x.shape[1], x.shape[2]]))
     
    # Second layer of LSTM
    model.add(Dense (units = 26))                 
   
     # last layer
    model.add(Dense(units = 1)) 
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model
def fit_modelControl(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)

    # shuffle = False because the order of the data matters
    historyControl = model.fit(Controle_X_train, Controle_Y_train, epochs = 20, validation_split = 0.2,
                    batch_size = 32, shuffle = False, callbacks = [early_stop])
    return historyControl

def controle(model):
    controle = model.predict(xtest)
    resultat_controle=np.delete(controle,np.s_[4:],axis=1)
    return resultat_controle
resultat_control=10*controle(reseau_control)  #forme reelle des element de sorties 
resultat_control=np.reshape(resultat_control,(1,4))
print(resultat_control[0][0])


