import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None)  

model = load_model('rf_fingerprinting_CNN_model3.keras')
label_encoder_identification = np.load(r"C:\Rf_data\label_encoder.npy")

def load_test_samples(h5_file_path):
    spectrograms = []
    psds = []
    sample_names = []

    with h5py.File(h5_file_path, 'r') as h5f:
        for test_name in h5f:
            test_group = h5f[test_name]
            for sample_name in test_group:
                sample_group = test_group[sample_name]
                spectrograms.append(sample_group['spectrograms'][()])
                psds.append(sample_group['psds'][()])
                sample_names.append(f"{test_name}/{sample_name}")

    return np.array(spectrograms), np.array(psds), sample_names

X_spectrogram, X_psd, sample_names = load_test_samples(r"C:\Rf_data\rf_fingerprintstestdata.h5")

predictions = model.predict([X_spectrogram, X_psd])
predicted_Device = np.argmax(predictions, axis=1)
confidence = np.max(predictions, axis=1) * 100

results = pd.DataFrame({
    'Sample': sample_names,
    'Device': [label_encoder_identification[DeviceID] for DeviceID in predicted_Device],
    'Predicted Device ': predicted_Device,
    'Confidence (%)': confidence
})

print("Results")
print(results)
