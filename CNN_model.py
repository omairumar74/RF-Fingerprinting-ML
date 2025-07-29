from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,BatchNormalization, Flatten, Dense, Dropout,Concatenate, ReLU, GlobalAveragePooling2D, GlobalAveragePooling1D

from tensorflow.keras.regularizers import l2

def build_combined_cnn_model(spectrogram_shape, psd_shape, num_classes):
 

    spectrogram_input = Input(shape=spectrogram_shape, name="spectrograms_input")
    x1 = Conv2D(8, kernel_size=3, padding='same')(spectrogram_input)
    x1 = ReLU()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)
    x1 = Conv2D(16, kernel_size=5, padding='same')(x1)
    x1 = ReLU()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)
    x1 = Conv2D(32, kernel_size=5, padding='same')(x1)
    x1 = ReLU()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling2D(pool_size=2)(x1)

    x1 = Flatten()(x1)


    psd_input = Input(shape=psd_shape, name="psds_input")
    x2 = Conv1D(8, kernel_size=3, padding='same')(psd_input)
    x2 = ReLU()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)

    x2 = Conv1D(16, kernel_size=5, padding='same')(x2)
    x2 = ReLU()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)


    x2 = Flatten()(x2)
  
    combined = Concatenate()([x1, x2])
    x = Dense(64, activation='relu', )(combined)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', )(combined)
    x = Dropout(0.2)(x)
    
    


    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[spectrogram_input, psd_input], outputs=outputs)

    model.summary()
    return model
