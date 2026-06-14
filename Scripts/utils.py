# Copyright (c) 2024, Icaro Nascimento Queiroz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy import array
import tensorflow as tf
from tensorflow.keras import layers, losses, regularizers, callbacks, models, Model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import optuna
import visualkeras
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler

class FixedCoeffLayer(tf.keras.layers.Layer):
    def __init__(self, coeffs, **kwargs):
        super().__init__(**kwargs)
        self.coeffs = tf.constant(coeffs, dtype=tf.float32)

    def build(self, input_shape):
        self.C = self.add_weight(
            name="fixed_coeffs",
            shape=(1, 1, 1, len(self.coeffs)),
            initializer=tf.constant_initializer(self.coeffs.numpy()),
            trainable=False
        )

    def call(self, inputs):
        return inputs * self.C

class OptFiltLayer(layers.Layer):
    def __init__(self, ai=None, bi=None, **kwargs):
        # super().__init__(trainable=False, **kwargs)
        super().__init__(**kwargs)
        self.trainable = False
        if ai is None:
            ai = [0.3594009, 0.49297974, 0.38133506, 0.24622458]
        if bi is None:
            bi = [-18.92073871, 0.90162148, 14.33011022, 6.34564695]
        self.ai = tf.constant(ai, dtype=tf.float32)
        self.bi = tf.constant(bi, dtype=tf.float32)
        self.nSamp = tf.shape(self.ai)[0]

    def call(self, inputs):
        rank = tf.rank(inputs)
        def from_flat():
            return tf.reshape(inputs, (-1, 49, self.nSamp))
        def from_reshaped():
            return inputs
        x = tf.cond(tf.equal(rank, 2), from_flat, from_reshaped)
        amp = tf.tensordot(x, self.ai, axes=[[2], [0]])
        time = tf.tensordot(x, self.bi, axes=[[2], [0]]) / amp 

        return time
    
    def get_config(self):
        return super().get_config()

class ConvDenoisingAutoencoderV3:
    def __init__(self,
                 input_shape=(7, 7, 1), 
                 filters=(18, 38),
                 kernel_size=(4, 4),
                 activation_func='relu',
                 padding_mode='same',
                 max_pooling=2,
                 up_sampling=2,
                 optimizing_func='adam',
                 loss_func='mse',
                 validation_split=0.1,
                 batch_size=256,
                 epochs=20):

        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation_func = activation_func
        self.padding_mode = padding_mode
        self.max_pooling = max_pooling
        self.up_sampling = up_sampling
        self.optimizing_func = optimizing_func
        self.loss_func = loss_func
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs

        self.fixed_coeffs = tf.constant([-18.92073871, 0.90162148, 14.33011022, 6.34564695], dtype=tf.float32)
        self.coeff = np.array([-18.92073871, 0.90162148, 14.33011022, 6.34564695])
        self.ai = [0.3594009, 0.49297974, 0.38133506, 0.24622458]
        self.bi = [-18.92073871, 0.90162148, 14.33011022, 6.34564695]
        self.autoencoder = self.build_model()


    def build_model(self):
        inp = layers.Input(shape=(7, 7, 4), name="stacked_frames")

        x = inp
        # x = layers.LayerNormalization(axis=[1,2,3])(inp * self.coeff.reshape(1, 1, 4))

        x = layers.Conv2D(self.filters[0], self.kernel_size, activation=self.activation_func, padding=self.padding_mode)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation_func)(x)
        x = layers.MaxPooling2D(self.max_pooling, padding=self.padding_mode)(x)

        x = layers.Conv2D(self.filters[1], self.kernel_size, activation=self.activation_func, padding=self.padding_mode)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation_func)(x)
        x = layers.MaxPooling2D(self.max_pooling, padding=self.padding_mode)(x)

        x = layers.Conv2DTranspose(self.filters[1], self.kernel_size, activation=self.activation_func, padding=self.padding_mode, strides=2)(x)
        x = layers.Activation(self.activation_func)(x)
        
        x = layers.Conv2DTranspose(self.filters[0], self.kernel_size, activation=self.activation_func, padding=self.padding_mode, strides=2)(x)
        x = layers.Activation(self.activation_func)(x)
        
        target_h, target_w = 7, 7
        x = layers.Resizing(target_h, target_w, interpolation="bilinear")(x)
        decoded = layers.Conv2DTranspose(4, (4, 4), padding=self.padding_mode, activation="linear", strides=1, name="decoded_frames")(x)
        reshaped = layers.Reshape((49, 4), name="reshape_49_4")(decoded)

        time_output = OptFiltLayer(ai=self.ai, bi=self.bi, name="optfilt_time_only")(reshaped)
        time_output = layers.LayerNormalization(axis=-1)(time_output)
        
        self.model = models.Model(inputs=inp, outputs=time_output)
        self.model.compile(optimizer=self.optimizing_func, loss=self.loss_func)
        self.model.summary()


    def train(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
        initial_loss = self.model.evaluate(
            X_train, y_train, verbose=0
        )
        initial_val_loss = None
        if X_val is not None and y_val is not None:
            initial_val_loss = self.model.evaluate(X_val, y_val, verbose=0)

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            callbacks=callbacks
        )
        history.history["loss"] = [initial_loss] + history.history["loss"]

        if initial_val_loss is not None:
            history.history["val_loss"] = [initial_val_loss] + history.history["val_loss"]

        return history

    def predict(self, X, scaler_x, scaler_y, batch_size=256):
        X = X.astype("float32")
        X_flat = X.reshape(len(X), -1)
        X_norm = scaler_x.transform(X_flat).reshape(len(X), 7, 7, 4)
        X_norm = X_norm.reshape(len(X), 7, 7, 4)
        pred_norm = self.model.predict(X_norm, batch_size=batch_size, verbose=0)
        pred = scaler_y.inverse_transform(pred_norm)
        return pred
    
    def summary(self):
        return self.model.summary()
    
    def optimize_with_optuna(X_train, y_train, X_val, y_val, scaler_x, scaler_y, n_trials=20, max_epochs=50):
        def objective(trial):
            filters = ( trial.suggest_int('filters_1', 64, 86),
                        trial.suggest_int('filters_2', 86, 128))

            kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (4, 4)])

            activation_func = trial.suggest_categorical('activation_func', ['relu', 'tanh'])

            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 3e-3)

            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

            up_sampling = trial.suggest_int('up_sampling', 2, 5)

            max_pooling = trial.suggest_int('max_pooling', 2, 5)
            model = ConvDenoisingAutoencoderV3(filters=filters,
                kernel_size=kernel_size,
                activation_func=activation_func,
                optimizing_func=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                    ),
                validation_split=0.1,
                batch_size=batch_size,
                epochs=max_epochs,
                up_sampling=up_sampling,
                max_pooling=max_pooling
            )
            history = model.model.fit(X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=25,
                            batch_size=256,
                            verbose=0,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=5,
                                restore_best_weights=True
                            )
                            ]
                        )
            return history.history['val_loss'][-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print("\n===== RESULTADOS OPTUNA =====")
        print(f"Melhor val_loss: {study.best_value:.6f}")
        print("Melhores hiperparâmetros encontrados:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=================================")
        return study

class ConvDenoisingAutoencoderV2: 
    def __init__(self, input_shape = (7, 7, 1), 
                 filters = (18, 38), 
                 kernel_size = (3, 3), 
                 activation_func = 'relu', 
                 padding_mode = 'same', 
                 max_polling = 2, 
                 up_sampling = 2, 
                 optimizing_func = 'adam', 
                 loss_func = 'mse', 
                 epochs_ = 10, 
                 batch_size_ = 256, 
                 validation_split_ = 0.2):
        
        self.input_shape = input_shape 
        self.filters = filters 
        self.kernel_size = kernel_size 
        self.activation_func = activation_func 
        self.padding_mode = padding_mode 
        self.max_polling = max_polling 
        self.up_sampling = up_sampling 
        self.optimizing_func = optimizing_func 
        self.loss_func = loss_func 
        self.epochs_ = epochs_ 
        self.batch_size_ = batch_size_ 
        self.validation_split_ = validation_split_ 
        self.autoencoder = self.build_model() 
        
    def build_model(self): 
        """Constrói a arquitetura do Denoising Autoencoder usando Conv2D.""" 
        # Encoder 
        encoder_input = tf.keras.Input(shape=self.input_shape) 
        x = layers.Conv2D(self.filters[0], self.kernel_size, padding=self.padding_mode)(encoder_input) 
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(self.activation_func)(x) 
        x = layers.MaxPooling2D((self.max_polling, self.max_polling), padding=self.padding_mode)(x)

        x = layers.Conv2D(self.filters[1], self.kernel_size, padding=self.padding_mode)(x) 
        x = layers.BatchNormalization()(x) 
        x = layers.Activation(self.activation_func)(x) 

        encoded_output = x 

        x = layers.Conv2DTranspose(self.filters[1], self.kernel_size, padding=self.padding_mode)(encoded_output) 
        x = layers.Activation(self.activation_func)(x) 

        x = layers.UpSampling2D((self.up_sampling, self.up_sampling))(x) 
        x = layers.Conv2DTranspose(1, self.kernel_size, padding=self.padding_mode)(x) 
        x = layers.BatchNormalization()(x)

        input_h, input_w = self.input_shape[0], self.input_shape[1] 
        output_shape = tf.keras.backend.int_shape(x) 
        out_h, out_w = output_shape[1], output_shape[2]

        if out_h > input_h or out_w > input_w: 
            crop_h = out_h - input_h 
            crop_w = out_w - input_w 
            x = layers.Cropping2D(cropping=((0, crop_h), (0, crop_w)))(x) 
        elif out_h < input_h or out_w < input_w: 
            pad_h = input_h - out_h 
            pad_w = input_w - out_w 
            x = layers.ZeroPadding2D(padding=((0, pad_h), (0, pad_w)))(x) 
            
        decoder_output = layers.Activation(self.activation_func)(x)
        self.model = models.Model(inputs=encoder_input ,outputs=decoder_output)
        self.model.compile(optimizer=self.optimizing_func, loss=self.loss_func)
        self.model.summary() 
    
    def train(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
        initial_loss = self.model.evaluate(
            X_train, y_train, verbose=0
        )
        initial_val_loss = self.model.evaluate(
            X_val, y_val, verbose=0
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_split=self.validation_split_,
            batch_size=self.batch_size_,
            epochs=self.epochs_,
            shuffle=True,
            callbacks=callbacks
        )

        history.history['loss'] = [initial_loss] + history.history['loss']
        history.history['val_loss'] = [initial_val_loss] + history.history['val_loss']

        return history
    
    def predict(self, X, scaler_x, scaler_y, batch_size=256):
        X = np.array(X).astype("float32")

        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)

        assert X.shape[1:] == (7,7,1), \
            f"Esperado shape (7,7,1), recebido {X.shape[1:]}"
        n_samples = len(X)
        X_flat = X.reshape(n_samples, -1)
        X_norm = scaler_x.transform(X_flat)
        X_norm = X_norm.reshape(n_samples, 7, 7, 1)
        pred_norm = self.model.predict(
            X_norm,
            batch_size=batch_size,
            verbose=0
        )
        pred_norm_flat = pred_norm.reshape(n_samples, -1)
        pred = scaler_y.inverse_transform(pred_norm_flat)
        pred = pred.reshape(n_samples, 7, 7, 1)

        return pred
    
    def summary(self):
        return self.model.summary()

    def optimize_with_optuna(X_train, y_train, X_val, y_val, scaler_x, scaler_y, n_trials=20, max_epochs=50):
        
        def normalize(X, scaler):
            return scaler.transform(X.reshape(len(X), -1)).reshape(len(X), 7,7,1)
        
        def objective(trial):
            filters = (
                trial.suggest_int('filters_1', 64, 86),
                trial.suggest_int('filters_2', 86, 128)
            )
            kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (4,4)])

            activation_func = trial.suggest_categorical('activation_func', ['relu', 'tanh', 'sigmoid'])

            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 3e-3)

            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

            up_sampling = trial.suggest_int('up_sampling', 2, 5)

            max_polling = trial.suggest_int('max_polling', 2, 5)

            X_train_norm = normalize(X_train, scaler_x)
            y_train_norm = normalize(y_train, scaler_y)

            X_val_norm = normalize(X_val, scaler_x)
            y_val_norm = normalize(y_val, scaler_y)

            model = ConvDenoisingAutoencoderV2(
                filters=filters,
                kernel_size=kernel_size,
                activation_func=activation_func,
                optimizing_func=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                batch_size_=batch_size,
                validation_split_=0.1,
                epochs_=max_epochs,
                up_sampling=up_sampling,
                max_polling=max_polling
            )
            
            history = model.model.fit(X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=25,
                            batch_size=256,
                            verbose=0,
                            callbacks=[
                                tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=5,
                                restore_best_weights=True
                            )
                            ]
                        )
            return history.history['val_loss'][-1]
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print("\n===== RESULTADOS OPTUNA =====")
        print(f"Melhor val_loss: {study.best_value:.6f}")
        print("Melhores hiperparâmetros encontrados:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("=================================")
        
        return study

def scatter_cells_R2(A, B, grid_shape=(7,7)):
    if A.ndim == 3:
        A = A.reshape(A.shape[0], -1)
        B = B.reshape(B.shape[0], -1)
    
    n_cells = A.shape[1]
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()
    r2_values = []

    for i in range(n_cells):
        ax = axes[i]
        y_true = B[:, i]
        y_pred = A[:, i]
        r2 = r2_score(y_true, y_pred)
        r2_values.append(r2)
        ax.scatter(y_true, y_pred, s=3)
        ax.set_title(f'Cell {i} — R²={r2:.3f}')
        ax.set_xlabel("Real")
        ax.set_ylabel("Predito")

    plt.tight_layout()
    plt.show()

    return np.array(r2_values)

def scatter_cells_R2_plus(A, B, grid_shape=(7,7), show_grid=True):
    if A.ndim == 3:
        A = A.reshape(A.shape[0], -1)
        B = B.reshape(B.shape[0], -1)

    n_cells = A.shape[1]
    rows, cols = grid_shape

    fig, axes = plt.subplots(rows, cols, figsize=(22, 22))
    axes = axes.flatten()

    r2_values = []
    rmse_values = []
    var_values = []
    baseline_pred = np.tile(B.mean(axis=0), (B.shape[0], 1))
    r2_baseline = r2_score(B.flatten(), baseline_pred.flatten())

    for i in range(n_cells):
        ax = axes[i]

        y_true = B[:, i]
        y_pred = A[:, i]

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        var = np.var(y_true)

        r2_values.append(r2)
        rmse_values.append(rmse)
        var_values.append(var)

        ax.scatter(y_true, y_pred, s=3, alpha=0.5)

        minv = min(y_true.min(), y_pred.min())
        maxv = max(y_true.max(), y_pred.max())
        ax.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)

        ax.set_title(f'Cell {i}\nR²={r2:.3f} | RMSE={rmse:.3f}')

        if show_grid:
            ax.grid(alpha=0.3)

    plt.suptitle(f"R² por célula — baseline R² = {r2_baseline:.3f}", fontsize=18)
    plt.tight_layout()
    plt.show()

    return {
        "r2_per_cell": np.array(r2_values),
        "rmse_per_cell": np.array(rmse_values),
        "var_per_cell": np.array(var_values),
        "r2_baseline": r2_baseline
    }

def jet_white():
    """
    Retorna um colormap jet modificado com a primeira cor branca.
    """
    cmap = plt.cm.jet  # original
    colors = cmap(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]  # primeira cor = branco
    return LinearSegmentedColormap.from_list("jet_white", colors)

def heatmap_all_cells(A, B, bins=300, cmap="jet_white", xlim=None, ylim=None):
    """
    Plota 49 heatmaps hist2d com fundo branco e colormap jet modificado.
    """

    # cria o cmap jet com fundo branco
    if cmap == "jet_white":
        cmap = jet_white()

    fig, axes = plt.subplots(7, 7, figsize=(16, 16))
    axes = axes.flatten()

    fig.patch.set_facecolor("white")

    for i, ax in enumerate(axes):
        y_true = B[:, i]
        y_pred = A[:, i]

    
        ax.set_facecolor("white")

        ax.hist2d(
            y_true, y_pred,
            bins=bins,
            cmap=cmap,
            range=[xlim, ylim] if (xlim and ylim) else None
        )
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
        ax.set_title(f"Célula {i+1}")

        r2 = r2_score(y_true, y_pred)

        err = y_pred - y_true
        bias = np.mean(err)
        std = np.std(err)

        ax.text(
            0.05, 0.83,
            f"R² = {r2:.3f}\nvar = {std:+.3f}",
            transform=ax.transAxes,
            fontsize=7,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    plt.tight_layout()
    plt.show()


def heatmap_all_cellsV2(A, B, bins=300, cmap="jet_white", xlim=None, ylim=None):
    """
    Plota 49 heatmaps hist2d com nomenclatura espacial das células.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    plt.rcParams.update({
    "font.size": 16,          # fonte padrão
    "axes.titlesize": 14,     # título dos subplots
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
    })
    # cria o cmap jet com fundo branco
    if cmap == "jet_white":
        cmap = jet_white()

    fig, axes = plt.subplots(7, 7, figsize=(16, 16))
    axes = axes.flatten()

    fig.patch.set_facecolor("white")

    # gera os labels no formato:
    # (-3,3), (-2,3), ..., (3,-3)
    labels = []

    for y in range(3, -4, -1):
        for x in range(-3, 4):
            labels.append(f"{x},{y}")

    for i, ax in enumerate(axes):

        y_true = B[:, i]
        y_pred = A[:, i]

        ax.set_facecolor("white")
        
        ax.hist2d(
            y_true,
            y_pred,
            bins=bins,
            cmap=cmap,
            range=[xlim, ylim] if (xlim and ylim) else None
        )

        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

        # título usando coordenadas da matriz
        ax.set_title(f"Célula ({labels[i]})")

        r2 = r2_score(y_true, y_pred)

        err = y_pred - y_true
        bias = np.mean(err)
        std = np.std(err)

        ax.text(
            0.05,
            0.83,
            f"R² = {r2:.3f}\nvar = {std:+.3f}",
            transform=ax.transAxes,
            fontsize=10.5,
            bbox=dict(
                facecolor='white',
                alpha=1,
                edgecolor='none'
            )
        )

    plt.tight_layout()
    plt.show()    

def percentual_error(matrix1, matrix2):
    absolute_difference = np.abs(matrix1 - matrix2)
    reference_value = matrix1 + matrix2
    reference_value[reference_value == 0] = 1
    percent_error = (absolute_difference / reference_value) * 100
    return percent_error

def reconstruction_error(original_data, reconstructed_data):
    '''
	This function calculates the RSE (Reconstruction Error)"
	'''
    return np.mean(np.square(original_data - reconstructed_data))

def mean_squared_error(original_data, reconstructed_data):
    '''
	This function calculates the MSE (Mean Squared Error)"
	'''
    return np.mean(np.square(original_data - reconstructed_data))

def mean_absolute_error(original_data, reconstructed_data):
    '''
	This function calculates the MAE (Mean Absolute Error)"
	'''
    return np.mean(np.abs(original_data - reconstructed_data))

def bits_per_pixel(original_data, reconstructed_data):
    '''
	This function calculates the BPP (Bits Per Pixel)"
	'''
    bpp = -np.log2(np.mean(np.exp(-np.square(original_data - reconstructed_data))))
    return bpp

def compare_autoencoder_error(X, Y, metric):
    '''
	This function calculates de MAE (Mean Absolute Error)"
	'''
    results = []
    for i in range(len(X)):
        input_matrix = X[i]
        output_matrix = Y[i]
        result = metric(input_matrix, output_matrix)
        results.append(result)
    return results

def compare_autoencoder_output(X, Y, metric):
    results = []
    for i in range(len(X)):
        input_matrix = X[i].reshape((7, 7))  # Reshape para matriz 7x7
        output_matrix = Y[i].reshape((7, 7))  # Reshape para matriz 7x7
        result = metric(input_matrix, output_matrix)
        results.append(result)
    return results

def getIdxClus_mxn(cluster, m, n):
    row, col = cluster.shape[0] // 2, cluster.shape[1] // 2    
    idx_mxn = cluster[row - m // 2:row + m // 2 + 1, col - n // 2:col + n // 2 + 1]
    
    return idx_mxn.flatten()

def OptFilt(samples, ai=None, bi=None):
    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
    idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
    ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
           '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
           '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
           '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
           '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
           '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
           '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]
    if samples.shape[1] == 25 : ij_cell = ij_cell[idx_5x5]
    if samples.shape[1] == 9 : ij_cell = ij_cell[idx_3x3]
    if ai is None:        
        # ai = [  0.36308142, 0.47002328, 0.39304565,  0.30191008]
        # bi = [-20.77449385, 5.48756441, 6.21710107, 10.33539619]
        
        ## Mykola coeffs
        ai = [0.3594009, 0.49297974, 0.38133506, 0.24622458]
        bi = [-18.92073871, 0.90162148, 14.33011022, 6.34564695]
        
    nSamp = len(ai)
    signals = int(samples.shape[1]/nSamp)
    AmpTime = dict()          
    
    AmpRec  = np.tensordot(samples.reshape(samples.shape[0], signals, nSamp),ai, axes=(2,0))
    TimeRec = (np.tensordot(samples.reshape(samples.shape[0], signals, nSamp), bi, axes=(2,0)))/AmpRec

    AmpTime = {'Cells': { f'Cell {ij_cell[i]}':{'Amplitude':AmpRec[:,i], 'StdAmp':np.std(AmpRec[:,i]), 'Time': TimeRec[:,i], 'StdTime':np.std(TimeRec[:,i])} for i in range(signals)}}
    
    AmpTime.update({'Clusters': {'Std': {'Amp': np.sum(AmpRec, axis=0).std(), 'Time':TimeRec.mean(axis=0).std()}, 
                                 'Mean':{ 'Time':TimeRec.mean(axis=0),'Amp': AmpRec.mean(axis=0)},
                                 'SumAmplitudes':AmpRec.sum(axis=1), 'Amplitude':AmpRec,'Time':TimeRec, 'RawData': samples}})
    return AmpTime

def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def min_max(x1, x2=None, x3=None):
    minx1 = min(x1)
    maxx1 = max(x1)

    if x2 is None: 
        minx2 = 1e10
        maxx2 = -1e10
    else : 
        minx2 = float('inf')
        maxx2 = -float('inf')

    if x3 is None: 
        minx3 = 1e10
        maxx3 = -1e10
    else : 
        minx3 = float('inf')
        maxx3 = -float('inf')

    return min([minx1, minx2, minx3]), max([maxx1, maxx2, maxx3])

def getMeanRms(x1, **kwargs):
    x2      = kwargs.get('x2')
    x3      = kwargs.get('x3')
    MeanRms = kwargs.get('MeanRms')
    
    if MeanRms.lower() == 'mean':
        if x2 is None and x3 is None: 
            return [np.mean(x1), np.std(x1)]
        elif x3 is None: 
            return [np.mean(x1), np.std(x1)], [np.mean(x2), np.std(x2)]
        else:
            return [np.mean(x1), np.std(x1)], [np.mean(x2), np.std(x2)], [np.mean(x3), np.std(x3)]
        
    if MeanRms.lower() == 'rms':
        if x2 is None and x3 is None: 
            return [rms(x1), rms(x1)]
        elif x3 is None: 
            return [rms(x1), np.std(x1)], [rms(x2), np.std(x2)]    
        else:
            return [rms(x1), np.std(x1)], [rms(x2), np.std(x2)], [rms(x3), np.std(x3)]

## Function to plot Histograms
def plotHisto(y1, y2=None, y3=None, **kwargs):
    """
    pathOut   - where you want to store the plot
    label     - is a list with 4 informations: 1st, 2nd, 3rd is the label for the signal, and the 4th is the legend position
    legend    - a list with symbol/name of the variable to show in histogram
    fileName  - File name to store
    titleName - Plot title
    show      - True sows the plot, False none
    log       - True ylog scale, False none
    ext       - File extension
    MeanRms:  
              - 'mean' to calculate mean and std values for each signal => mean +/- std
              - 'rms' to calculate the rms and rmse values for each signal => rms +/- rmse
    save      - True save plot on pathOut, false none
    y1        - first signal  (mlp)
    y2        - second signal (of)
    y3        - third signal (target)
    
    """    
    # Define default values for parameters
    default_params = {
        "axisFormat": True,        
        "adjustXlim": True,
        "detail": False,
        "ext": "png",
        "fileName": "histo",
        "label": "",
        "legend": ['','',''],
        "log": False,
        "MeanRms": "mean",
        "pathOut": "",
        "save": False,        
        "show": True,
        "titleName": "",
        "text": "",
        "unit": "",
        "xRange": None,
        
    }
    def custom_formatter(x, pos):
        threshold = 0.001  # Threshold for scaling
        scale_x = 10000
        if abs(x) < threshold:
            return '{:.2f}'.format(x * scale_x)  # Format the scaled value to 3 decimal places
        else:
            magnitude = len(str(int(x))) - 1  # Calculate the magnitude of the number
            decimals = max(0, 2 - magnitude)   # Determine the number of decimal places
            return '{:.{dec}f}'.format(x, dec=decimals)  # Format with the determined number of decimal places
        # Update default parameters with provided keyword arguments
    params = {**default_params, **kwargs}

    plt.rcParams["figure.figsize"] = (10,6)
    plt.rcParams.update({
    'font.size': 22,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 22,
    })

    scale_x = 1    
    xscale = 1
    
    if params["label"].lower() == 'energy':
        if y2 is not None and y3 is not None: 
            if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
                y1, y2, y3 = y1*1e3, y2*1e3, y3*1e3
                xLabel = r'Energia [$10^{-3}$'+f' {params["unit"]}]'
            else :
                xLabel = f'Energia [{params["unit"]}]'
        elif y2 is not None:
            if rms(y1) < 0.01 or rms(y2) < 0.01:
                y1, y2 = y1*1e3, y2*1e3                
                xLabel = r'Energia [$10^{-3}$'+f' {params["unit"]}]'
            else :
                xLabel = f'Energy [{params["unit"]}]'
        else:
            if rms(y1) < 0.01:
                y1 = y1*1e3
                xLabel = r'Energia [$10^{-3}$'+f' {params["unit"]}]'
            else :
                xLabel = f'Energia [{params["unit"]}]'
    elif params["label"] == 'time':        
        xLabel = 'Tempo [ns]'
        unit   = 'ns'
    elif params["label"] == 'ss':
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''        
    else :
        if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
            xscale  = 10000
            xLabel = f'{params["label"]} '+r'[$10^{-4}$]'  
        else: 
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''                
        
    Nbins, Ndpi  = 100, 700
    if params["xRange"]:        
        minBin, maxBin = min(params["xRange"]), max(params["xRange"])
    else :
        minBin, maxBin = min_max(y1, y2, y3)
    valueBins = np.linspace(minBin, maxBin, Nbins)
    
    ## ===========================
    ## ---------- Details --------
    #if detail is True:
    unit    = params["unit"]
    if params["detail"]:
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [12, 8]})
        maxStd = maxBin
        meanMean = np.mean([y1, y2, y3])
        y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
        ## ==========================
        ## ==============
        ## THREE plots
        #n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{params["legend"][0]}'.ljust(10,' ')+f'rms: {rms(y1):.2f} $\pm$ {rmsErr(y1):.2f} {unit}'.ljust(28,' '))
        n, bins, patches = ax[0].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{params["legend"][0]}'.ljust(10,' ')+ f': {y1MeanRms[0]*xscale:.2f} $\pm$ {y1MeanRms[1]*xscale:.2f} '.ljust(20,' '))
        ax[0].hist(y2,  bins=valueBins,  ec='red',  alpha=0.5, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{params["legend"][1]}'.ljust(14,' ')+ f': {y2MeanRms[0]*xscale:.2f} $\pm$ {y2MeanRms[1]*xscale:.2f} '.ljust(20,' '))
        ax[0].hist(y3,  bins=valueBins,  ec='k',    alpha=0.4, fc='gainsboro', lw=1.8, histtype='stepfilled', label=f'{params["legend"][2]}'.ljust(14,' ')+ f': {y3MeanRms[0]*xscale:.2f} $\pm$ {y3MeanRms[1]*xscale:.2f} '.ljust(20,' '))
    
        xmin,xmax,ymin,ymax = ax[0].axis()
        ax[0].legend(frameon=False, fontsize=30, loc='upper left', bbox_to_anchor=(1.02, 1.0))

        if log == True:
            ax[0].set_yscale('log')
            ax[0].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[0].set_ylim([None, 1.4*ymax])
            ax[0].set_ylabel(f'Count'+r' [10$^3$]')    
            
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[0].yaxis.set_major_formatter(ticks_y)
        else: ax[0].set_ylabel('Count')            
            
        ax[0].set_xlabel(f'{xLabel}')         
        ax[0].grid(ls='--', lw=0.7)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        
        if adjustXlim is True:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[0].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[0].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[0].set_xlim([bins[ii-5], bins[ii+5]])
                    
        ## ==============
        ## TWO plots
        n, bins, patches = ax[1].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{params["legend"][0]}')
        ax[1].hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.6, histtype='stepfilled', label=f'{params["legend"][2]}')
        xmin,xmax,ymin,ymax = ax[1].axis()
        
        nn = np.where(n>10)
        #if adjustXlim is True:
        if params["adjustXlim"]:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[1].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[1].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[1].set_xlim([bins[ii-5], bins[ii+5]])
        
        ax[1].legend(frameon=False)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        #if log == True:
        if params["log"]:            
            ax[1].set_yscale('log')
            ax[1].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[1].set_ylim([None, 1.4*ymax])
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[1].yaxis.set_major_formatter(ticks_y)
            
        #ax[1].axes.yaxis.set_visible(False)
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{xLabel}') 
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)        
             
    ## ===========================
    ## ---- Whithout Details -----
    else :        
        fig    = plt.figure()
        ax     = fig.add_subplot(111)
        MeanRms = params["MeanRms"]
        titleName = params["titleName"]
        if (y2 is not None and y3 is not None):
            
            y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2), np.std(y3))
            meanMean = np.mean([y1, y2, y3])            
            
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{params["legend"][0]}'.ljust(10,' ')+ f'{y1MeanRms[0]*xscale:.2f} $\pm$ {y1MeanRms[1]*xscale:.2f}'.rjust(20,' '))
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{params["legend"][1]}'.ljust(10,' ')+ f'{y2MeanRms[0]*xscale:.2f} $\pm$ {y2MeanRms[1]*xscale:.2f} '.rjust(20,' '))
            ax.hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.5, histtype='stepfilled', label=f'{params["legend"][2]}'.ljust(10,' ')+ f' {y3MeanRms[0]*xscale:.2f} $\pm$ {y3MeanRms[1]*xscale:.2f} '.rjust(20,' '))

            nn = np.where(n>10)
                ## --------------- ##
        ## ==========================
        ## ==============
        ## TWO plots
        elif (y2 is not None):
            y1MeanRms, y2MeanRms = getMeanRms(y1, x2=y2, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2))
            meanMean = np.mean([y1, y2])        

            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{params["legend"][0]}'.ljust(11,' ')+ f' {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f}'.rjust(20,' '))
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled',                    label=f'{params["legend"][1]}'.ljust(11,' ')+ f' {y2MeanRms[0]:.2f} $\pm$ {y2MeanRms[1]:.2f}'.rjust(20,' '))

        ## ==========================
        ## ==============
        ## ONE plot
        else :
            y1MeanRms = getMeanRms(y1, MeanRms=MeanRms)
            maxStd = np.std(y1)
            meanMean = np.mean(y1)
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.7, histtype='stepfilled', label=f'{params["legend"][0]}'.ljust(10,' ')+ f' {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} '.ljust(20,' '))
        
        ax.set_title(titleName)
        ax.set_xlabel(f'{xLabel}')
        ax.grid(linestyle='--',linewidth=.7)
        #ax.legend(frameon=False, loc=legend[3])
        ax.legend(frameon=False, loc='upper right')
        # ax.legend(frameon=False)

        xmin,xmax,ymin,ymax = ax.axis()
        nn = np.where(n>5)
        #if log == True:
        if params["log"]:
            plt.gca().set_yscale("log")
            #ax.set_ylim([None, 40*abs(ymax)])
            ax.set_ylim([100, 8*abs(ymax)])
            ax.set_ylabel(f'Count'+r' [10$^3$]')        
            scale_y = 1e3
            fileName = f'{fileName}_log'
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count'+r' [10$^3$]')
        else :
            scale_y = 1
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count')
            
        limit = 0.0
        #if adjustXlim is True:
        if params["adjustXlim"]:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax.set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax.set_xlim([bins[0], bins[ii+5]])
                else :
                    ax.set_xlim([bins[ii-5], bins[ii+5]])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
        
    if params["text"]:
        #textstr = '\n'.join((
        #f'E$_{{imp}}$',
        #f'{params["text"]}'))
        textstr = f'\n {params["text"]}'
        # plt.text(1, 6.5, f'{params["text"]}', fontsize=12)#, color='red')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
        verticalalignment='top')
        
    if params["save"]:
        fig.savefig(f'{params["pathOut"]}/{params["fileName"]}.{params["ext"]}', format=params["ext"], dpi=Ndpi)
    #print(f'Show plot: {show}')

    if not params["show"]:
    #if show == False :
        plt.close(fig)
        plt.close()
    #elif show == True:
    elif params["show"]:
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()
    del fig   