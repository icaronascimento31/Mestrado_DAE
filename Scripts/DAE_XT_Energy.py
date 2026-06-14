#!/usr/bin/python3
import sys
import os

sys.path.append(os.path.abspath(".."))

import pickle
import Scripts.utils
from Scripts.utils import *
import numpy as np
from numpy import array
import glob
import pandas as pd
import matplotlib.pyplot as plt
import random, math
import matplotlib.ticker as ticker
import seaborn as sns
from rich.console import Console
from rich.table import Table
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from tensorflow.keras.utils import plot_model
import visualkeras
from PIL import ImageFont, Image
from sklearn.metrics import r2_score

import copy
import random, math
import time
from joblib import dump
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from sklearn.model_selection import train_test_split
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

np.seterr(divide='ignore', invalid='ignore')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

dataSamplesGev = np.load('../Data/lzt_dictClus7x7_99k_SignalsSamples_e50_GeV.pkl', allow_pickle=True)
dataAmplitudeGev = np.load('../Data/lzt_dictClus7x7_99k_Amplitudes_e50_GeV.pkl', allow_pickle=True)

# Leitura de elementos do sinal do vetor de amostras
EtruthSampGeV = dataSamplesGev['E'] 
XTcSampGeV = dataSamplesGev['XT_C']
XTlSampGeV = dataSamplesGev['XT_L']
XTrSampGeV = dataSamplesGev['XT_R']
NoiseSampGeV = dataSamplesGev['Noise']

# Sinal de leitura de energia somado as contribuições de Crosstalk e ruído
xDataSamplesGeV = np.add(np.add(np.add(np.add(EtruthSampGeV, XTcSampGeV), XTlSampGeV), NoiseSampGeV), XTrSampGeV)

# Leitura de elementos do sinal do vetor de amplitudes
EtruthAmpGeV = dataAmplitudeGev['E']
XTcAmpGeV = dataAmplitudeGev['XT_C']
XTlAmpGeV = dataAmplitudeGev['XT_L']
XTrAmpGeV = dataAmplitudeGev['XT_R']
NoiseAmpGeV = dataAmplitudeGev['Noise']

# Sinal de leitura de energia somado as contribuições de Crosstalk e ruído
xDataAmpGeV = np.add(np.add(np.add(np.add(EtruthAmpGeV, XTcAmpGeV), XTlAmpGeV), NoiseAmpGeV), XTrAmpGeV)

AmpTimeGeV  = OptFilt(EtruthSampGeV)
AmpTimeXTGeV  = OptFilt(xDataSamplesGeV)

AmplitudesGeVOptFilt = AmpTimeGeV['Clusters']['Amplitude']
XTAmplitudesGeVOptFilt = AmpTimeXTGeV['Clusters']['Amplitude']

shape_val = xDataSamplesGeV.shape[0]

TimesGeVOptFilt = AmpTimeGeV['Clusters']['Time']

scaler_x_data = MinMaxScaler()

xDataSamplesGeV_flat = xDataSamplesGeV.reshape((shape_val, -1))
xDataSamplesGeV_Normalized = scaler_x_data.fit_transform(xDataSamplesGeV_flat)
xDataSamplesGeV_Normalized = xDataSamplesGeV_Normalized.reshape((shape_val,7,7,4))


scaler_data_amp = MinMaxScaler()
scaler_etruth = MinMaxScaler()
scaler_time = MinMaxScaler()
scaler_samp = MinMaxScaler()
scaler_data_samp = MinMaxScaler()
xDataAmpGeV_Normalized = scaler_data_amp.fit_transform(xDataAmpGeV)
EtruthAmpGeV_Normalized = scaler_etruth.fit_transform(EtruthAmpGeV)
TimesGeVOptFilt_Normalized = scaler_time.fit_transform(TimesGeVOptFilt)


xDataSampGeV_Normalized = scaler_data_samp.fit_transform(xDataSamplesGeV)
EtruthSampGeV_Normalized = scaler_samp.fit_transform(EtruthSampGeV)

toGeV = 1000

ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
           '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
           '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
           '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
           '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
           '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
           '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]



Scaled_EtruthSampGeV = EtruthSampGeV.astype('float32')/toGeV
Scaled_xDataSampGeV_Normalized = xDataSampGeV_Normalized.astype('float32')

E_truth_energy_toConv = Scaled_EtruthSampGeV.reshape(shape_val * 4, 7, 7, 1)
XTData_denoise_toConv = xDataSampGeV_Normalized.reshape(shape_val * 4, 7, 7, 1)

X = XTData_denoise_toConv           
y = E_truth_energy_toConv  

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=42
)

def scale_X(X, scaler):
    Xn = scaler.transform(X.reshape(len(X), -1))
    return Xn.reshape(len(X), 7, 7, 1)

def scale_y(y, scaler):
    return scaler.transform(y.reshape(len(y), -1))

conv_Denoise = ConvDenoisingAutoencoderV2(input_shape=(7, 7, 1), 
                                        filters=(64, 86),
                                        kernel_size=(4, 4),
                                        activation_func = 'relu',
                                        padding_mode = 'same',
                                        max_polling = 2,
                                        up_sampling = 2,
                                        optimizing_func = 'adam',
                                        loss_func = 'mse',
                                        epochs_ = 20,
                                        batch_size_ = 256,
                                        validation_split_ = 0.2
                                        )
Denoise_Network = conv_Denoise.train(X_train, y_train, X_val, y_val)

y_true_real = scaler_time.inverse_transform(
    E_truth_energy_toConv.reshape(len(E_truth_energy_toConv), -1)
)

y_pred_real = scaler_time.inverse_transform(
    Denoise_Network.model.predict(XTData_denoise_toConv).reshape(len(E_truth_energy_toConv), -1)
)

r2_real = r2_score(y_true_real, y_pred_real)

heatmap_all_cells(y_true_real, y_pred_real)

rmse_cells = []

for i in range(y_true_real.shape[1]):
    rmse_i = np.sqrt(
        mean_squared_error(
            y_true_real[:, i],
            y_pred_real[:, i]
        )
    )
    rmse_cells.append(rmse_i)

rmse_cells = np.array(rmse_cells)
rmse_map = rmse_cells.reshape(7, 7)

rmse_cells = np.array(rmse_cells)  # (49,)

fig, axes = plt.subplots(7, 7, figsize=(10, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.set_facecolor("white")

    # escreve o RMSE no centro da célula
    ax.text(
        0.5, 0.5,
        f"{rmse_cells[i]:.3f}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_title(f"Célula {i+1}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)

plt.suptitle("RMSE por célula (matriz 7×7)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()