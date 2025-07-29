import json
import numpy as np
import h5py
from scipy.signal import spectrogram, welch, get_window


input_folder = r'C:\Rf_data'  
output_h5 = r'C:\Rf_data\rf_fingerprints.h5'
fs = 2.048e6  
n_samples = 100  
win_len = 1024  
noverlap = 512  


window = get_window('hann', win_len)

with h5py.File(output_h5, 'w') as RF_Data_group:
    for device_id in range(1, 11):
        meta_path = f"{input_folder}\\Device {device_id}\\Device{device_id}.sigmf-meta"
        data_path = f"{input_folder}\\Device {device_id}\\Device{device_id}.sigmf-data"
        
        with open(meta_path, 'r') as serialdata:
            meta = json.load(serialdata)
        serial_number = meta['global']['core:description']  
        
        iq = np.fromfile(data_path, dtype=np.int16)
        iq = iq[::2] + 1j * iq[1::2] 
        
        chunk_size = len(iq) // n_samples
        
        group = RF_Data_group.create_group(serial_number)
                            
        for sample_number in range(n_samples):
            start = sample_number * chunk_size
            end = start + chunk_size
            chunk = iq[start:end]
            
            f, t, s = spectrogram(chunk, fs=fs, nperseg=win_len, noverlap=noverlap, window=window)
            spectrograms = 10 * np.log10(np.abs(s) + 1e-12)
            
            freq, psds = welch(chunk, fs=fs, nperseg=win_len, noverlap=noverlap, window=window)
            psds = 10 * np.log10(psds + 1e-12)

            
            spectrograms = (spectrograms - np.min(spectrograms)) / (np.max(spectrograms) - np.min(spectrograms))
            psds = (psds - np.min(psds)) / (np.max(psds) - np.min(psds))
            
            spectrograms = np.expand_dims(spectrograms, axis=-1) 
            psds = np.expand_dims(psds, axis=-1)


            processed_rf_data = group.create_group(f"samples{sample_number}")
            processed_rf_data.create_dataset('spectrograms', data=spectrograms)
            processed_rf_data.create_dataset('psds', data=psds)
            processed_rf_data.attrs['device_id'] = device_id
            
