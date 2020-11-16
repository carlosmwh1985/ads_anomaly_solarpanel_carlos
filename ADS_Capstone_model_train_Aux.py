# Advanced Data Science - Capstone Project

## Functions Definitions - To Help in the Model Training

### Carlos Granados

# Standard libraries:
import numpy as np
import pandas as pd
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import Callback

#### Some Functions used in the Model Training:

# Function to extract times in a given time-interval

def divideTimeSteps(df, dt, nDay=0, nMonth=0):
    df_t = []
    nMonth_min = int(df['MONTH'].min())
    nMonth_max = int(df['MONTH'].max()) + 1
    if nMonth != 0:
        nMonth_min = nMonth
        nMonth_max = nMonth + 1
    for nm in range(nMonth_min, nMonth_max):
        df0 = df[df['MONTH'] == nm]
        nDay_min = int(df0['DAY'].min())
        nDay_max = int(df0['DAY'].max()) + 1
        if nDay != 0:
            nDay_max = nDay_min + nDay + 1
        for nd in range(nDay_min, nDay_max):
            df1 = df0[df0['DAY'] == nd]
            t_min = df1['TIME'].min()
            t_max = df1['TIME'].max()
            nt = ceil((t_max - t_min) / dt)
            t0 = t_min
            t1 = t_min
            for i in range(nt):
                t1 += dt
                temp = df1[(df1['TIME'] >= t0) & (df1['TIME'] < t1)]
                df_t.append(temp)
                t0 = t1
    return df_t

# Function to make all data pre-processing, as defined in the 
#  feature engineering phase.
def dataProcess(df_gen, df_sen, df_t, n, dt=4.0, nDay=0, nMonth=0):
    """
    Function to make ALL the data pre-processing:
      ETL, data cleasing, feature engineering...
    VARIABLES:
    df_gen : DF containing the generated power data
    df_sen : DF containing the sensor data for the same plant
    df_t   : DF containing the complete set of days and times
    n      : Number of sources to be extracted
    return : Array of DFs with the n processed DFs, for each source...
    """
    # Column Names to be changed
    dic1 = {"AMBIENT_TEMPERATURE": "AMB_TEMP"}
    dic2 = {"MODULE_TEMPERATURE": "MOD_TEMP"}
    # Columns to drop at the end
    cols_drop = ['DATE_TIME', 'PLANT_ID_x', 'SOURCE_KEY_x',
                 'SOURCE_KEY_y', 'PLANT_ID_y', 'TOTAL_YIELD',
                 'DAILY_YIELD', 'DT_TIME']
    # Columns to be considered at the final DF
    cols_end = ['TIME', 'DAY', 'MONTH', 'AMB_TEMP', 'MOD_TEMP',
                'IRRADIATION', 'AC_POWER', 'DC_POWER']
    # Start with transforming the datetime object to a valid format
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'])
    df_sen['DATE_TIME'] = pd.to_datetime(df_sen['DATE_TIME'])
    # Obtain all different sources
    df_sub = df_gen['SOURCE_KEY'].value_counts().reset_index()
    n = min(n, df_sub.shape[0])
    dfn_all = {}
    dfn_temp = {}
    # Extract the DF for each DF and merge it with the sensor data
    for i in range(n):
        src = df_sub['index'][i]
        print('Extracting data for Source Key : ' + src)
        df = df_gen[df_gen['SOURCE_KEY'] == src]
        df = df.merge(df_sen, on='DATE_TIME', how='left')
        dfn_temp[i] = df
    # Add MONT, DT_TIME and TIME, and change the name of the Temperature Cols.
    for i in range(n):
        dfi = dfn_temp[i]
        # Add columns related with time (MONTH, DAY, DT_TIME (as datetime Obj.)
        #   and TIME as a float)
        dfi['MONTH'] = pd.DatetimeIndex(dfi['DATE_TIME']).month
        dfi['DAY'] = pd.DatetimeIndex(dfi['DATE_TIME']).day
        dfi['DT_TIME'] = pd.to_datetime(dfi['DATE_TIME'])
        dfi['TIME'] = dfi['DT_TIME'].dt.hour + dfi['DT_TIME'].dt.minute/60.0
        # Change column names related to the temperature
        dfi.rename(columns=dic1, inplace=True)
        dfi.rename(columns=dic2, inplace=True)
        # Merge with the complet set of times and dates, and replace NaNs
        dfi = df_t.merge(dfi, on=['TIME', 'DAY', 'MONTH'], how='left')
        dfi.fillna(0, inplace=True)
        # Drop columns
        dfi.drop(cols_drop, axis=1, inplace=True)
        # Change the order of the features
        dfi = dfi[cols_end]
        # Store the modified DF
        dfn_temp[i] = dfi
    # Create subsets, for given periods of time
    icount = 0
    for i in dfn_temp:
        dfi = dfn_temp[i]
        temp = divideTimeSteps(dfi, dt, nDay=nDay, nMonth=nMonth)
        for dfj in temp:
            dfn_all[icount] = dfj
            icount += 1        
    return dfn_all

# Custom class, to help define the normalization of the data set, according
#  to the values present in the data set

class customNorm(object):
    """
    Class to define the normalization of data,
    associated with the power generation of
    solar panels
    """
    
    def __init__(self, df_gen, df_sen, max_AmbTemp=40.0, max_ModTemp=70.0,
                 max_I=None, max_DC=None, max_AC=None):
        # Copy variables to class vars
        self.df_gen = df_gen.copy(deep=True)
        self.df_sen = df_sen.copy(deep=True)
        self.max_AmbTemp = max_AmbTemp
        self.max_ModTemp = max_ModTemp
        self.max_I = max_I
        self.max_DC = max_DC
        self.max_AC = max_AC
        
        # Defined norms for time: hour, day month
        self.norm_t = 1.0 / 24.0
        self.norm_d = 1.0 / 31.0
        self.norm_m = 1.0 / 12.0
        
        # Extract Max. values of enviromental variables,
        # in particular solar irradiation
        if self.max_AmbTemp == None:
            self.max_AmbTemp = self.df_sen['AMBIENT_TEMPERATURE'].max()
        if self.max_ModTemp == None:
            self.max_ModTemp = self.df_sen['MODULE_TEMPERATURE'].max()
        if self.max_I == None:
            self.max_I = self.df_sen['IRRADIATION'].max()
        
        self.norm_AmbTemp = 1.0 / self.max_AmbTemp
        self.norm_ModTemp = 1.0 / self.max_ModTemp
        self.norm_I = 1.0 / self.max_I
        
        # Extract Max. values of AC/DC power generation
        if max_DC == None:
            self.max_DC = self.df_gen['DC_POWER'].max()
        if max_AC == None:
            self.max_AC = self.df_gen['AC_POWER'].max()
            
        self.norm_AC = 1.0 / self.max_AC
        self.norm_DC = 1.0 / self.max_DC
        
    def MaxNorm(self, df):
        """
        Custom scaler...
        df : (DF) Data to be normalized
        """
        #self.df = df
        self.temp = df.copy(deep=True)
        # Normalize time related features
        self.temp.loc[:,'TIME'] *= self.norm_t
        self.temp.loc[:,'DAY'] *= self.norm_d
        self.temp.loc[:,'MONTH'] *= self.norm_m
        
        # Normalize enviromental variables
        self.temp.loc[:,'IRRADIATION'] *= self.norm_I
        self.temp.loc[:,'AMB_TEMP'] *= self.norm_AmbTemp
        self.temp.loc[:,'MOD_TEMP'] *= self.norm_ModTemp
        
        # Normalize power values
        self.temp.loc[:,'DC_POWER'] *= self.norm_DC
        self.temp.loc[:,'AC_POWER'] *= self.norm_AC
    
        # Transform DF to a numpy array
        self.data = self.temp.to_numpy()
        
        return self.data

    def NoNorm(self, df):
        """
        Only transform the DF to a numpy array...
        df : (DF) Data to be normalized
        """
        #self.df = df
        self.temp = df.copy(deep=True)
    
        # Transform DF to a numpy array
        self.data = self.temp.to_numpy()
        
        return self.data
    
    def get_df(self):
        return self.df
    
    def get_MaxTemp(self):
        return [self.max_AmbTemp, self.max_ModTemp]

# Function to trimm and normalize the data

def create_trimmed_data_norm(df, steps, scaler):
    dim = df.shape[-1]
    samples = df.shape[0]
    #data = df.to_numpy()
    #samples = len(data)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #data = scaler.fit_transform(data)
    data = scaler.MaxNorm(df)
    #data = scaler.NoNorm(df)
    trim = samples % steps
    recording_trimmed = data[:samples-trim]
    recording_trimmed.shape = (int((samples-trim)/steps), steps, dim)
    return recording_trimmed

# Function to plot the loss and the distribution of its values...

def plotLoss(dataLoss, loss, opt):
    """
    Function to plot the losses given after the
    training process.
    dataLoss : (DF or np.array), loss data
    loss     : (str) name of the used loss function
    opt      : (str) name of the used optimizer
    """
    gblue = '#1565C0'   #blue 800
    gred = '#E53935'    #red 600
    ggreen = '#2E7D32'  #green 800
    gorange = '#E64A19' #deep orange 700
    fig, ax = plt.subplots(figsize=(10,5))
    ls_all = ['-', '--', ':', '-.']
    col_all = [gblue, gred, ggreen, gorange]
    df0 = dataLoss
    n = len(df0)
    ax.plot(range(1, n+1), df0, c=gorange)
    ax.set_xlabel('epochs')
    ax.set_ylabel('Loss')
    fig.suptitle('Model using Loss : ' + loss + ', and Optimizer : ' + opt)
    plt.show()
    
def distLoss(dataLoss, loss, opt, bins=20, alpha=0.85):
    """
    Function to plot the distribution of the losses given
    after the training process.
    dataLoss : (DF or np.array), loss data
    loss     : (str) name of the used loss function
    opt      : (str) name of the used optimizer
    bins     : (int) number of bins in the Dist. plot
    alpha    : (float) alpha of distribution plot
    """
    gblue = '#1565C0'   #blue 800
    gred = '#E53935'    #red 600
    ggreen = '#2E7D32'  #green 800
    gorange = '#E64A19' #deep orange 700
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set_style(style='whitegrid', rc={'figure.figsize' : (100,10)})
    df0 = np.array(dataLoss).T
    df0 = pd.DataFrame(data=dataLoss, columns=['Loss'])
    sns.displot(data=df0, x='Loss', bins=bins, color=gorange, alpha=alpha,
                height=5, aspect=2)
    plt.show()

# Modification to the Callbacks, in order to save the losses and the accuracy

losses = []

def handleLoss(loss):
    global losses
    losses += [loss]
    print(loss)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        handleLoss(logs.get('loss'))

# Function to train a given model

def trainModel(df_data, model, lossHistory, epochs, lossFun, optFun,
               actFun, batch_size, time_steps, scaler, dataTrimming,
               testModel, nNeurons=50, nLayers=10, nLoops=1, init=1):
    """
    Function to train a model, for a giving data set
    df_data      : (df), contains the data ued to train the model
    model        : Null if it will be used for the 1st time, a traned model otherwise
    epochs       : (int), number of epochs
    nLoops       : (int), number of times to repeat the training. 1 by default
    lossFun      : array (Str), loss function to be used
    optFun       : array (str), optmizer to be used
    actFun       : array (str), activation function to be used
    batch_size   : (int)
    timesteps    : (int)
    dataTrimming : Function to be used to reshape the data set
    scaler       : (Obj) initialized scaler to renormalized the data
    testModel    : Function that gives the model to be used
    init         : (int), function to define if a session must be restarted or not. 1 by default
    """
    # Get reshaped data (normalization, if needed, must be done in this function)
    data = dataTrimming(df_data, time_steps, scaler)
    # Clear session, to start with a blank state on each function call
    if init == 1:
        tf.keras.backend.clear_session()
        # Initialize model, compile it and train it...
        model = testModel(time_steps, dim, actFun, nNeurons=nNeurons, nLayers=nLayers)
        model.compile(optimizer=optFun, loss=lossFun, metrics=['accuracy'])
    # Array to store all the final losses and accuracy tests, at each loop
    lossHistory = LossHistory()
    lossVals = []
    for i in range(nLoops):
        model.fit(data, data, epochs=epochs, batch_size=batch_size,
                  validation_data=(data, data), shuffle=False, verbose=0,
                  callbacks=[lossHistory])
        lossVals.append(lossHistory.losses)
    flat = [x for sublist in lossVals for x in sublist]
    return [model, flat]
