# Advanced Data Science - Capstone Project

## Functions Definitions - To Help in the Model Definition

### Carlos Granados

# Standard libraries:
import numpy as np
import pandas as pd
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns

#### Some Functions used in the Model Definition:

def create_trimmed_data(df, steps):
    dim = df.shape[-1]
    data = df.to_numpy()
    samples = len(data)
    trim = samples % steps
    recording_trimmed = data[:samples-trim]
    recording_trimmed.shape = (int((samples-trim)/timesteps), timesteps, dim)
    return recording_trimmed

def create_trimmed_data_norm(df, steps):
    dim = df.shape[-1]
    data = df.to_numpy()
    samples = len(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    trim = samples % steps
    recording_trimmed = data[:samples-trim]
    recording_trimmed.shape = (int((samples-trim)/timesteps), timesteps, dim)
    return recording_trimmed
    
#And a function to plot the accuracy and the losses...

def plotLossAcc(dataLoss, dataAcc, loss, opt, labels):
    gblue   = '#1565C0' #blue 800
    gred    = '#E53935' #red 600
    ggreen  = '#2E7D32' #green 800
    gorange = '#E64A19' #deep orange 700
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ls_all = ['-', '--', ':', '-.']
    col_all = [gblue, gred, ggreen, gorange]
    for i in range(4):
        df0, df1 = dataLoss[i], dataAcc[i]
        n = len(df0)
        ax[0].plot(range(1, n+1), df0, label=labels[i], ls=ls_all[i], c=col_all[i])
        ax[1].plot(range(1, n+1), df1, label=labels[i], ls=ls_all[i], c=col_all[i])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('epochs')
    ax[1].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy')
    fig.suptitle('Model using Loss : ' + loss + ', and Optimizer : ' + opt)
    plt.show()

#And we define a function to automatize a bit the training

def lossExplorer(losses, opts, acts, df_data, dataTrimming, testModel):
    """
    Function to explore a set of loss, optmizers and activation functions,
    for a giving model...
    losses       : array (Str), containing all the loss function to be evaluated
    opts         : array (str), with the optmizers to be evaluated
    acts         : array (str), with the activation functions to be evaluated
    df_data      : (df), contains the data ued to train the model
    dataTrimming : Function to be used to reshape the data sets
    testModel    : Function that gives the model to be evaluated
    Parameters that must be defined before to call this functions:
    epochs       : (int)
    timesteps    : (int)
    batch_size   : (int)
    """
    # Array to store all the final losses and accuracy tests, at each loop
    resEval = []
    # Loop on the losses and the optimizers
    for loss in losses:
        print('Loss : ', loss)
        for opt in optimizers:
            print('Optimizer : ', opt)
            lossVals = []
            accVals = []
            for act in actFuns:
                #print('Activation Function : ', act)
                # Initialize array to record losses and accuracy...
                lossHistory = LossHistory()
                # Get data
                data = dataTrimming(df_data, timesteps)
                # Clear session, to start with a blank state on each loop
                tf.keras.backend.clear_session()
                # Initialize model, compile it and train it...
                model = testModel(data, timesteps, dim, act)
                model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
                model.fit(data, data, epochs=epochs, batch_size=batch_size,
                          validation_data=(data, data), shuffle=False, verbose=0,
                          callbacks=[lossHistory])
                #print(lossHistory.losses)
                # Recover Losses and Accuracies for each epoch
                lossVals.append(lossHistory.losses)
                accVals.append(lossHistory.acc)
                resEval.append((loss, opt, act, lossVals[-1], accVals[-1]))
            # Plot Loss and Accuracy for all activation functions, for a fixed Opt. and Loss.
            print(np.array(lossVals).shape)
            plotLossAcc(dataLoss=lossVals, dataAcc=accVals,
                        loss=loss, opt=opt, labels=actFuns)
    #At the end, return the array with all results!
    return resEval

#Function to transform the evaluation array obtained from
# the lossExplorer funtion to a data frame
#(loss, opt, act, lossVals[-1], accVals[-1])
def eval2df(evalData, cols):
    temp0 = []
    temp1 = []
    dim = len(cols)
    iCount = 0
    loss = []
    acc = []
    a = np.array(['Loss-Opt.'])
    b = np.array(cols)
    colNames = np.append(a, b)
    print(colNames)
    for val in evalData:
        test = val[0] + ' - ' + val[1]
        fun = val[2]
        if iCount == 0:
            loss.append(test)
            acc.append(test)
        if fun != cols[iCount]:
            print('Some error with function name!')
            break
        x0 = val[3][-1]
        x1 = val[4][-1]
        loss.append(x0)
        acc.append(x1)
        if iCount == dim-1:
            temp0.append(loss)
            temp1.append(acc)
            #print(temp)
            iCount = -1
            loss = []
            acc = []
        iCount += 1
    temp0 = np.array(temp0)
    temp1 = np.array(temp1)
    df_temp0 = pd.DataFrame(data=temp0, columns=colNames)
    df_temp1 = pd.DataFrame(data=temp1, columns=colNames)
    return [df_temp0, df_temp1]

# Some colors, usign Google Material Design Color Palette
gblue = '#1565C0'   #blue 800
gred = '#E53935'    #red 600
ggreen = '#2E7D32'  #green 800
gorange = '#E64A19' #deep orange 700

def plotVars(df, var1, var2, lab_all, x_lab, y_lab, title, xnorm=0, ynorm=0, nslices=0):
    """
    Function to plot several properties of a data frame...
    It asumes that in the x-axis always the same variable (var1) is used,
    while on the y-axis it must be given in an array...
    VARS:
    df      : Data Frame
    var1    : string
    var2    : string (Array)
    lab_all : string (Array)
    norm    : To normalize to 1 all variables. By defect it doesn't make it
    nslices : To consider the first rows of df, if it != 0
    """
    fig, ax = plt.subplots(figsize=(12,8))
    n = len(var2)
    ls_all = ['-', '--', ':', '-.']
    col_all = [gblue, gred, ggreen, gorange]
    xmax = 1.0
    ymax = 1.0
    if nslices != 0:
        df = df.head(nslices)
    if xnorm != 0:
        xmax = df[var1].max()
    for i in range(n):
        if ynorm != 0:
            ymax = df[var2[i]].max()
        ax.plot(df[var1]/xmax, df[var2[i]]/ymax, label=lab_all[i],
                ls=ls_all[i], c=col_all[i])
    ax.legend()
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)

def plotCorr(df, cols):
    """
    Function to plot the correlation matrix as a heatmap
    df : Data Frame
    cols: array of columns to consider
    """
    corr = df[cols].corr()
    fig = plt.subplots(figsize=(10, 5))
    ax = sns.heatmap(corr, #vmin=-1, vmax=1,
                     center=0,
                     #cmap=sns.diverging_palette(20, 220, n=200),
                     square=True)
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       horizontalalignment='right')
