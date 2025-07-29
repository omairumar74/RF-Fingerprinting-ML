import tensorflow as tf
from CNN_model import build_combined_cnn_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def load_data(h5_path):
    spectrograms = []
    psds = []
    device_ids = []
    
    with h5py.File(h5_path, 'r') as file:
        for device_name in file.keys():
            device_group = file[device_name]
            for sample_name in device_group.keys():
                sample_group = device_group[sample_name]
                spectrograms.append(sample_group['spectrograms'][()])
                psds.append(sample_group['psds'][()])
                device_ids.append(sample_group.attrs['device_id'])  
    return np.array(spectrograms), np.array(psds), np.array(device_ids)


spectrograms, psds, device_ids = load_data(r'C:\Rf_data\rf_fingerprints.h5')

label = LabelEncoder()
encoded_labels = label.fit_transform(device_ids)  

X_spec_train, X_spec_test, X_psd_train, X_psd_test, y_train, y_test = train_test_split(
    spectrograms, psds, encoded_labels, test_size=0.1, random_state=30
)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


model = build_combined_cnn_model(
    spectrogram_shape=X_spec_train.shape[1:],
    psd_shape=X_psd_train.shape[1:],
    num_classes=10
)

model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='categorical_crossentropy',
    metrics=['accuracy','Precision', 'Recall']
)

history = model.fit(
    [X_spec_train, X_psd_train], y_train,
    validation_data=([X_spec_test, X_psd_test], y_test),
    epochs=30,  
    batch_size=16,
   
)

test_loss, test_acc, test_precision, test_recall = model.evaluate([X_spec_test, X_psd_test], y_test)
print(f"Accuracy: {test_acc:%}, Precision: {test_precision:%}, Recall: {test_recall:%}")


model.save('rf_fingerprinting_CNN_model3.keras')

np.save('label_device.npy', encoded_labels)
