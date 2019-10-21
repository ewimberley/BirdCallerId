#!/usr/bin/python3
import tensorflow as tf
from scipy.io.wavfile import read
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os

PATH_SEPARATOR = os.path.sep
TRAIN = "train"
VALIDATE = "validate"
ID = "id"
DATASET = "dataset"
SPECIES_ID = "speciesId"
DATA_PATH = "Data"


def wav_file_to_npy(filename):
    """Load .wav file from filename and return the sampling frequency and waveform in a numpy array."""
    wav = read(filename)
    wavNp = np.array(wav[1], dtype=float)
    return wav[0], wavNp


def sample_windows(sample_len_seconds, samples_per_minute, time, windows_per_sec, x):
    """Get equally spaced starting indices across the waveform."""
    num_windows = len(x)
    num_samples = int(samples_per_minute * time / 60.0)
    windows_per_sample = int(sample_len_seconds * windows_per_sec)
    sample_start_indices = np.linspace(0, num_windows - windows_per_sample, num=num_samples, dtype=np.int32)
    return sample_start_indices, windows_per_sample


def compute_species_sampling_ratio(df, dataset_name, id_to_sampling_freqs, id_to_waveforms):
    """Return a sub-sampling ratio for each species id in order to ensure equal class balance."""
    species_to_time = {}
    species_sampling_ratio = {}
    for index, row in df.iterrows():
        dataset = row[DATASET]
        if dataset == dataset_name:
            id = str(row[ID])
            freq = id_to_sampling_freqs[id]
            data = id_to_waveforms[id]
            time = float(np.shape(data)[0]) / float(freq)
            species_id = str(row[SPECIES_ID])
            if species_id not in species_to_time:
                species_to_time[species_id] = time
            else:
                species_to_time[species_id] = species_to_time[species_id] + time
    min_species_time = species_to_time["0"]
    for species_id in species_to_time:
        if species_to_time[species_id] < min_species_time:
            min_species_time = species_to_time[species_id]
    for species_id in species_to_time:
        species_sampling_ratio[species_id] = min_species_time / species_to_time[species_id]
    return species_sampling_ratio


def shuffle_data_and_labels(x, y):
    """Randomly shuffle two equal sized numpy arrays to the same order."""
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)


def create_dataset(data_file, sample_len_seconds, samples_per_minute):
    """Create a set of numpy arrays with the correct shape for input the the neural network."""
    X_train = []
    y_train = []
    X_validate = []
    y_validate = []

    id_to_sampling_freqs = {}
    id_to_waveforms = {}

    def get_dataset_arrays(dataset):
        x_array = X_train
        y_array = y_train
        if dataset == VALIDATE:
            x_array = X_validate
            y_array = y_validate
        return x_array, y_array

    # load sound data from disk
    df = pd.read_csv(data_file, sep="\t")
    for index, row in df.iterrows():
        id = str(row[ID])
        data_file = id + ".wav"
        freq, wave = wav_file_to_npy(DATA_PATH + PATH_SEPARATOR + data_file)
        id_to_sampling_freqs[id] = freq
        id_to_waveforms[id] = wave

    # Get an equal number of samples per species
    species_sampling_ratio = {}
    species_sampling_ratio[TRAIN] = compute_species_sampling_ratio(df, TRAIN, id_to_sampling_freqs, id_to_waveforms)
    species_sampling_ratio[VALIDATE] = compute_species_sampling_ratio(df, VALIDATE, id_to_sampling_freqs, id_to_waveforms)

    # process the samples
    print("File\tSpecies\tSpeciesId\tSampling Freq (Hz)\tLength (Secs)")
    samples_per_species = {}
    for index, row in df.iterrows():
        id = str(row[ID])
        species = str(row['species'])
        species_id = str(row[SPECIES_ID])
        dataset = row[DATASET]
        freq = id_to_sampling_freqs[id]
        wave = id_to_waveforms[id]
        time = float(np.shape(wave)[0]) / float(freq)
        seconds = (str(int(time)) + " Seconds")
        sampling_freq = str(freq) + " Hz"
        print(data_file + "\t" + species + "\t" + species_id + "\t" + sampling_freq + "\t" + seconds)
        all_sample_start_indices, windows_per_sample = sample_windows(sample_len_seconds, samples_per_minute, time, freq, wave)
        x_array, y_array = get_dataset_arrays(dataset)
        sample_start_indices = np.random.choice(all_sample_start_indices, int(
            species_sampling_ratio[dataset][species_id] * len(all_sample_start_indices)), replace=False)
        if dataset not in samples_per_species:
            samples_per_species[dataset] = {}
            samples_per_species[dataset][species_id] = len(sample_start_indices)
        else:
            if species_id not in samples_per_species[dataset]:
                samples_per_species[dataset][species_id] = len(sample_start_indices)
            else:
                samples_per_species[dataset][species_id] = samples_per_species[dataset][species_id] + len(sample_start_indices)
        for start_index in sample_start_indices:
            end_index = start_index + windows_per_sample
            sample = wave[start_index:end_index, ]
            x_array.append(sample)
            y_array.append(species_id)
    print("Training samples: " + str(samples_per_species[TRAIN]))
    print("Validation samples: " + str(samples_per_species[VALIDATE]))

    shuffle_data_and_labels(X_train, y_train)
    shuffle_data_and_labels(X_validate, y_validate)
    return np.stack(X_train), np.stack(y_train), np.stack(X_validate), np.stack(y_validate)


def add_channel_shape(x):
    """Shape numpy array for convolutional layer (requires a channel dimension)."""
    return np.reshape(x, (x.shape[0], x.shape[1], 1))


X_train, y_train, X_validate, y_validate = create_dataset("example_data.csv", 12.0, 225)
print("*" * 30)
num_classes = np.unique(y_train).shape[0]
X_train = add_channel_shape(X_train)
X_validate = add_channel_shape(X_validate)
y_train = to_categorical(y_train)
y_validate = to_categorical(y_validate)

ACTIVATION_FUNC = 'relu'
# TODO procedurally generate network structure
model = tf.keras.models.Sequential([
    tf.keras.layers.GaussianNoise(0.01),
    tf.keras.layers.Conv1D(128, kernel_size=(31), strides=(6), activation=ACTIVATION_FUNC,
                           input_shape=(X_train.shape[1], 1), data_format="channels_last"),
    tf.keras.layers.Conv1D(64, kernel_size=(21), strides=(4), activation=ACTIVATION_FUNC),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation=ACTIVATION_FUNC),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation=ACTIVATION_FUNC),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation=ACTIVATION_FUNC),
    tf.keras.layers.Conv1D(64, kernel_size=(11), strides=(4), activation=ACTIVATION_FUNC),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation=ACTIVATION_FUNC),
    tf.keras.layers.Dense(500, activation=ACTIVATION_FUNC),
    tf.keras.layers.Dense(250, activation=ACTIVATION_FUNC),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1.5e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_validate, y_validate), epochs=14)
print(model.summary())
loss, acc = model.evaluate(X_validate,
                           y_validate)  # , batch_size=BATCH_SIZE) #use batch size if GPU memory is getting low
print("Loss: " + str(loss))
print("Accuracy: " + str(acc))
y_pred = model.predict(X_validate)
matrix = confusion_matrix(y_validate.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
