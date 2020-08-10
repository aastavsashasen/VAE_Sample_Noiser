# For data handling (loading, processing)
from scipy.io import wavfile
import tensorflow as tf
import sounddevice
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
# For AI shit/ Autoencoder implementation
from keras.layers import Input, Dense
from keras.models import Model

audio_data_list = np.zeros(shape=(1,2))
audio_frames_list = np.array([])
print(audio_data_list)
fs_data_list = np.array([])

# =========================== PREPARING DATA
# Have to find a way to deal with different audio file lengths
# at first we can cut them all short (to the shortest).
# We can also introduce silence to the shorter files to extend them.

# Let us load and handle the data here to make it all the same length
# also fill silence if necessary
sample_length = 10000
default_fs = 44100

for filename in glob.glob('audio/*.wav'):  # Load Wav files from audio folder
    fs, data = wavfile.read(filename)
    data_cut = data[0:sample_length]
    if len(data_cut) < sample_length:  # add silence to short data files
        silence_to_add = sample_length - len(data_cut)
        data_cut = np.vstack((data_cut, np.zeros(shape=(silence_to_add,2))))
    audio_frames_list = np.append(audio_frames_list, data_cut.shape[0])
    audio_data_list = np.vstack((audio_data_list, data_cut))
    fs_data_list = np.append(fs_data_list, fs)

# will require a Auto Encoder to map a space of samples and allow us to inpaint/extrapolate the space
# ========================================= DATA HANDLING

fs1, data1 = wavfile.read('audio/AMBblondie.wav')
fs2, data2 = wavfile.read('audio/PERCtamb1.wav')
# Data = numpy array of #rows determined by sample
# consisting of y-axis ranges (2 cols) ie) [[],[],...]

print("list:", audio_data_list)  # numpy array of audio
print("shape of list: ", audio_data_list.shape)
final_audio_array = audio_data_list[1:audio_data_list.shape[0]]
print("shape of final array: ", final_audio_array.shape)
print("Final array of audio: ", final_audio_array)
print("data lengths: ", audio_frames_list)
print("total audio files: ", len(audio_frames_list))
print("FS data list: (sample rates): ", fs_data_list)  # commonly all 44100 Hz

plt.figure("sound1")
plt.plot(data1)
plt.figure("sound2")
plt.plot(audio_data_list)
#plt.show()

sounddevice.play(audio_data_list, fs1*1)  # releases GIL
time.sleep(4)
#sounddevice.play(data2, fs2)
#time.sleep(1)


# ===================================================== Creating our model (Keras)
# samples have different lengths (different input dimensions)
# We handled this by cutting/extending to a length of 20000
# Assume all introduced samples have a frame rate of 44100 Hz (conventional)

# this is the size of our encoded representations
encoding_dim = 1000  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
loss_array = np.array([])


for encoding_dim in range(1000, 10000, 1000):
    # changing filename to save output decoded samples
    filename = "wavTest" + str(encoding_dim) + ".wav"
    # this is our input placeholder
    input_img = Input(shape=(2*sample_length,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(2*sample_length, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # Now we load and flatten our data, (will all be the same length)
    x_flat = final_audio_array.flatten()
    print("Flattened shape: ", x_flat.shape)
    x_train = x_flat.reshape((len(audio_frames_list), 2*sample_length))
    print("Training data shape: ", x_train.shape)

    autoencoder.fit(x_train, x_train,
                    epochs=100,  # original = 50 epochs
                    batch_size=4,  # original 256
                    shuffle=True,
                    validation_data=(x_train, x_train))  # originally x_test

    # encode and decode some samples
    # should be from the test set
    encoded_samples = encoder.predict(x_train)
    decoded_samples = decoder.predict(encoded_samples)

    print("Shape of Encoded samples: ", encoded_samples.shape)
    print("Shape of Decoded samples: ", decoded_samples.shape)
    decoded_output = decoded_samples.reshape((sample_length*len(audio_frames_list), 2))
    print("decoded output shape: ", decoded_output.shape)

    # ====================== Playing the decoded samples...
    print("Compressed Dimensions: ", encoding_dim)
    print("ENCODED: ")
    sounddevice.play(final_audio_array, fs1*1.0)  # releases GIL
    time.sleep(4)
    print("DECODED: ")
    sounddevice.play(decoded_output, fs1*1.0)  # releases GIL
    time.sleep(4)
    # Save to disk:
    wavfile.write(filename, default_fs, decoded_output)













