# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:03:02 2024



#| This file handles the model creation for the DNN used for speech recognition, use name_attempt_distortion2sd.weights.h5 for the most 
#| accurate weightings

@author: tommy
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os
from scipy.interpolate import CubicSpline
import librosa
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2
from audio_extract import extract_audio
from main import applyFbankLogDCT, toMagFrames, getEnergy
from visual_preprocessing import detect_face, dct8by8, detect_mouth, get_frame, bitmap_smooth
names = {
    0 : "Muneeb",
    1 : "Zachary",
    2 : "Sebastian",
    3 : "Danny",
    4 : "Louis",
    5 : "Ben",
    6 : "Seb",
    7 : "Ryan",
    8 : "Krish",
    9 : "Christopher",
    10 : "Kaleb",
    11 : "Konark",
    12 : "Amelia",
    13 : "Emilija",
    14 : "Naima",
    15 : "Leo",
    16 : "Noah",
    17 : "Josh",
    18 : "Joey",
    19 : "Kacper"
    }


# #|Loading in all audio files found in directory
# for name in names:
#     no = 0
#     #print(name)
#     name = names[name]
#     for i, soundfile in enumerate(sorted(glob.glob(f'AVPSAMPLESSD\\{name}\\*.wav'))):
#
#         #r.reshape(self, shape)
#         speechFile, frequency = sf.read(soundfile, dtype='float32')
#         noise, freq = sf.read('noise.wav', dtype='float32')
#         #print(noise.shape)
#         print(speechFile.shape)
#         noise = noise[:speechFile.shape[0]]
#         #| Adding noise distortion
#         if no < 15:
#             noisePower = np.mean(noise**2)
#             speechPower = np.mean(speechFile**2)
#             amplification = np.sqrt((speechPower/noisePower)*(10**(-(10/10))))
#             speechFile = speechFile+(amplification*noise)
#             # sd.play(speechFile, 16000)
#             # sd.wait()
#         if no > 15 and no < 25:
#             noisePower = np.mean(noise**2)
#             speechPower = np.mean(speechFile**2)
#             amplification = np.sqrt((speechPower/noisePower)*(10**(-(20/10))))
#             speechFile = speechFile+(amplification*noise)
#             # sd.play(speechFile, 16000)
#             # sd.wait()
#
#         #print(soundfile)
#         #| Applying preprocessing feature extraction
#         mag_frames = toMagFrames(speechFile)
#         file_energy = getEnergy(speechFile)
#         mfccFile = applyFbankLogDCT(mag_frames, file_energy)
#         #| Saving mfccs by name and number
#         np.save(f'mfccs\\{name}\\{name}_{no}', mfccFile)
#         no = no + 1
# #
# Skeleton for the visual processing
#


def processVisualData():
    # for name in names:
    no = 0
    #     name = names[name]
    for vid in enumerate(sorted(glob.glob(r'C:\Users\tommy\Desktop\AudiovisualLabs\AVPSum\Christopher\*.mp4'))):
        print(vid)
        frames = get_frame(vid)
        vid_coefficients = []
        for frame in frames:
            faces = detect_face(frame)
            detected_mouth = detect_mouth(faces)
            c = dct8by8(detected_mouth)
            vid_coefficients.append(c)
            print(c.shape)

        np.save(f'Visualmfccs\\Christopher\\Christopher_{no}', vid_coefficients)
        no += 1
        # a = np.array(vid_coefficients)
        # print(a.shape)
        
def processMP4Data():
    for name in names:
        vid_no = 0
        audio_no = 0
        visual_no = 0
        name = names[name]
        for vid in enumerate(sorted(glob.glob(f'C:\\Users\\scott\\Documents\\Unreal Projects\\AVPSum\\mp4s\\{name}\\*.mp4'))):
            #goes through each vid and extracts the audio as .wav
            extract_audio(input_path=f"mp4s\\{vid}\\{vid}_{vid_no}.mp4", output_path=f"wavs\\{vid}\\{vid}_{vid_no}.wav", format= "wav")
            vid_no += 1
            
        for i, soundfile in enumerate(sorted(glob.glob(f"wavs\\{name}\\*.wav"))):
            speechFile_48k, frequency = sf.read(soundfile, dtype='float32')
            #frequency should be 48k here, resample to 16k
            speechFile_16k = librosa.resample(speechFile_48k, orig_sr= frequency, target_sr=16000)
        
            #| Applying preprocessing feature extraction
            mag_frames = toMagFrames(speechFile_16k)
            file_energy = getEnergy(speechFile_16k)
            mfccFile = applyFbankLogDCT(mag_frames, file_energy)
            
            #this is where we need to do the visual stuff so that we can append it to the npy file before saving
            #hopefully this then grabs the frames of the relevant mp4
            vis_frames = get_frame(f'\\mp4s\\{name}\\{name}_{audio_no}.mp4')
            #this returns an interpolated array where the visual frames line up to the mccFiles (audios)
            interp_vis = visual_feature_interp(mfccFile, vis_frames)
            #the frames in interp_vis are the ones we want to make into bitmaps and dct

            vis_coefficients = []
            vis_bitmaps = []
            for frame in interp_vis:
                faces = detect_face(frame)
                detected_mouth = detect_mouth(faces)
                c = dct8by8(detected_mouth)
                c2 = bitmap_smooth(detected_mouth)
                vis_coefficients.append(c)
                vis_bitmaps.append(c2)
                
            row_n = mfccFile.shape[0] #bottom row
            mfccFile = np.insert(mfccFile,row_n,vis_coefficients,axis=0)
            row_n = mfccFile.shape[0] #new bottom row
            mfccFile = np.insert(mfccFile,row_n,vis_bitmaps,axis=0)
            #so now we have our audio mfcc array with 2 extra rows: 1 with the dcts and 1 with bitmaps
            
            
            #| Saving mfccs by name and number
            np.save(f'mfccs\\{name}\\{name}_{audio_no}', mfccFile)
            audio_no += 1
        
processVisualData()
#| Data and label arrays
data = []
labels = []
i = 0
#| Calculating max length of mfcc file shapes
max_length1 = 0
max_length0 = 0

#|
#| This function is used to calculate the max length for dimension 0 and dimension 1
for mfcc_file in sorted(glob.glob('mfccs/*/*.npy')):
    mfcc_data = np.load(mfcc_file)
    if mfcc_data.shape[0] > max_length0:
        max_length0 = mfcc_data.shape[0] 
    if mfcc_data.shape[1] > max_length1:
        max_length1 = mfcc_data.shape[1]


## For each mfcc_file in mfcc folter
for mfcc_file in sorted(glob.glob('mfccs/*/*.npy')):
    ## load the mfcc data
    mfcc_data = np.load(mfcc_file)
    #print(mfcc_file)
    #| padding the mfcc data to ensure that the mfccs are the same dimensions, using the previously calculated 
    #| max lengths for both dimensions
    mfcc_data = np.pad(mfcc_data, ((0,max_length0-mfcc_data.shape[0]), (0, max_length1-mfcc_data.shape[1]))) 
    #| Appending this to training data
    data.append(mfcc_data)
    #print(mfcc_data.shape)
    #| Getting the file name
    stemFileName = (Path(os.path.basename(mfcc_file)).stem)
    #| Splitting the file by _ to get the name rather than number
    label = stemFileName.split('_')
    #print(label)
    #| Appending the name to array of labels to match the classes
    labels.append(label[0])

## Converting data into a numpy array
data = np.array(data)
labels = np.array(labels)


classes = []

## Appending all values in defined name dictionary to a classes array
for val in names.values():
    classes.append(val)

#|
#| Normalising each data index by its own highest value, this is to account for differing amplitudes per mfcc
for val in range(len(data)):
    data[val] = data[val] / np.max(data[val])
    
#|
#| Using One-Hot encoding, transforming labels into binary vectors for classes
LE = LabelEncoder()
LE = LE.fit(classes)
#print(classes)
labels = to_categorical(LE.transform(labels))  
#print(labels)

#| Training and testing split
X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0, stratify=labels, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5,  random_state=0, stratify=y_tmp, shuffle=True)

#|
#| Creating the DNN 
def createDNN():
    numClasses = 20
    model = Sequential()
    #| Shape to match padded mfccs
    model.add(InputLayer(shape=(max_length0,max_length1,1)))
    #| 5 layers, 64 kernel size
    #model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model

#|
#| Setting epochs and batch sizes
def train_model():
    #| 10 epochs provided stable loss decrease and accuracy increases
    num_epochs = 10
    num_batch_size = 32
    #| Compiling each time to avoid building upon previous weightings
    model = createDNN()
    #| Slower learning rate, reduced by factors of 10, to account for more layers and to reduce intial model loss
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
   
    
    model.summary()
    
    #| Fitting the model with training and validation data
    training = model.fit(X_train, y_train, validation_data=(X_val,
    y_val), batch_size=num_batch_size, epochs=num_epochs,
    verbose=1)
    #| Saving the weights to remove the need to completely retrain the model
    model.save_weights('woutpitch.weights.h5')

    #| Plotting the loss and the accuracy
    plt.figure()    
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    

#| This function is used to create the confusion matrix, based on the test and validation data preditions
def prediction():
    model = createDNN()
    model.load_weights('woutpitch.weights.h5')   
    predicted_probs=model.predict(X_test, verbose=0)
    predicted=np.argmax(predicted_probs,axis=1)
    #print(predicted.label)
    actual=np.argmax(y_test,axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print(f'Accuracy: {accuracy * 100}%')
    #| Plotting confusion matrix
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(y_test,axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
                                                confusion_matrix)
    cm_display.plot()

#| This function was created to take live audio recordings and get a prediction based off
#| the current model weightings
def test():
    #| Intialising model
    model = createDNN()
    
    #| Two second recording, mono
    speechFile = sd.rec(2 * 16000, samplerate=16000, channels=1)
    #| Letting me know if it has reached the time to record
    print("recording")
    sd.wait()
    sd.play(speechFile, 16000)
    print("Done!")
    
    #| Preprocessing the audio file
    mag_frames = toMagFrames(speechFile)
    file_energy = getEnergy(speechFile)
    mfcc_data = applyFbankLogDCT(mag_frames, file_energy)
    
    #| Adding padding so that it matches the input shape of the neural network
    mfcc_data = np.pad(mfcc_data, ((0,max_length0-mfcc_data.shape[0]), (0, max_length1-mfcc_data.shape[1]))) 
    
    #| Creating a numpy array to store sd.rec recorded audio
    testing_data = []
    testing_data.append(mfcc_data)
    testing_data = np.array(testing_data)
    print(testing_data.shape)
    #| Normalising each audio recording 
    for val in range(len(testing_data)):
        testing_data[val] = testing_data[val] / np.max(testing_data[val])
      
    #| Prediction
    predicted_name = model.predict(data, verbose=0)
    predicted_id = np.argmax(predicted_name, axis=1)
    predicted_class = LE.inverse_transform(predicted_id)
    print(predicted_class)
    confusion_matrix = metrics.confusion_matrix(
    np.argmax(y_test,axis=1), predicted_id)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
    confusion_matrix)
    cm_display.plot()
    
def visual_feature_interp(visual_feat, audio_feat):
    """
    Return visual features matching the number of frames of the supplied audio
    feature. The time dimension must be the first dim of each feature
    matrix; uses the cubic spline interpolation method - adapted from Matlab.

    Args:
        visual_feat: the input visual features size: (visual_num_frames, visual_feature_len)
        audio_feat: the input audio features size: (audio_num_frames, audio_feature_len)

    Returns:
        visual_feat_interp: visual features that match the number of frames of the audio feature
    """

    audio_timesteps = audio_feat.shape[0]

    # Initialize an array to store interpolated visual features
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[1]))

    for feature_dim in range(visual_feat.shape[1]):
        cubicSpline = CubicSpline(np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim])
        visual_feat_interp[:, feature_dim] = cubicSpline(np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps))

    return visual_feat_interp

def test_brute():

    model = createDNN()
    model.load_weights('new.weights.h5')   
   
    voice_test = []
    speechFile = sd.rec(2* 16000, samplerate=16000, channels=1, dtype='float32')
    
    print(speechFile.shape)
 
   
    print("recording")
    sd.wait()
    #sd.play(speechFile, 16000)
    print("Done!")
    #| Normalising speech based on how training data was normalised
    speechFile = 0.99 * speechFile / max(abs(speechFile))
    #| Preprocessing 
    speechFile = speechFile.squeeze()
    mag_frames = toMagFrames(speechFile)
    file_energy = getEnergy(speechFile)
    mfcc_data = applyFbankLogDCT(mag_frames, file_energy)
    plt.imshow(mfcc_data, origin='lower')
    plt.show()
    mfcc_data = np.pad(mfcc_data, ((0,max_length0-mfcc_data.shape[0]), (0, max_length1-mfcc_data.shape[1]))) 
    plt.imshow(mfcc_data, origin='lower')
    voice_test.append(mfcc_data)
    
    #print(data[0].shape)
    #print(voice_test[0].shape)
    # for data in range(len(test_data)):
    #     voice_test.append(test_data[data])
    voice_test = np.array(voice_test)
    for val in range(len(voice_test)):
        voice_test[val] = voice_test[val] / np.max(voice_test[val])
    predicted_name = model.predict(voice_test, verbose=0)
    predicted_id = np.argmax(predicted_name, axis=1)
    predicted_class = LE.inverse_transform(predicted_id)
 

    print(f'Predicted name {predicted_class}')

#train_model()
#test()
#test_brute()
#prediction()

#| This function was used to record and store names in different directories using sd.rec
def recording_test():
    #| For every name in dictionary names
    for name in names:
        #| Getting the actual name value by index, would otherwise be a number for the index of names
        name = names[name] 
        #| Making new child directory for each name
        os.mkdir(f'AVPSAMPLESSD\\{name}')
        #| Used for making a max of 29 samples per name
        i = 0     
        while i < 30:
            #| Debugging to let me know it was ready to say name for the ith time
            print(f'recording {name} {i}')   
            speechFile = sd.rec(2* 16000, samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            print("done")
            speechFile = 0.99 * speechFile / max(abs(speechFile))
            #| The if conditional is primarily used to keep consistency with other samples
            if i < 10:
                sf.write(f'AVPSAMPLESSD\\{name}\\{name}00{i}sd.wav', speechFile, 16000)
            else:
                sf.write(f'AVPSAMPLESSD\\{name}\\{name}0{i}sd.wav', speechFile, 16000)
            i = i +1
            


def testing_recordings():
    #| For every name in dictionary names
    for name in names:
    #     #| Getting the actual name value by index, would otherwise be a number for the index of names
        name = names[name] 
    #     #| Making new child directory for each name
    #     os.mkdir(f'AVPSAMPLESSD\\TEST\\{name}')
    #     #| Used for making a max of 29 samples per name
    #     i = 0     
    #     while i < 10:
    #         #| Debugging to let me know it was ready to say name for the ith time
    #         print(f'recording {name} {i}')   
    #         speechFile = sd.rec(2* 16000, samplerate=16000, channels=1, dtype='float32')
    #         sd.wait()
    #         print("done")
    #         speechFile = 0.99 * speechFile / max(abs(speechFile))
    #         #| The if conditional is primarily used to keep consistency with other samples
    #         if i < 10:
    #             sf.write(f'AVPSAMPLESSD\\TEST\\{name}\\{name}00{i}sd.wav', speechFile, 16000)
    #         else:
    #             sf.write(f'AVPSAMPLESSD\\TEST\\{name}\\{name}0{i}sd.wav', speechFile, 16000)
    #         i = i +1
            
        no = 0
        for file in sorted(glob.glob(f'AVPSAMPLESSD\\TEST\\{name}\\*.wav')):
            speechFile, frequency = sf.read(file, dtype='float32')
 
            noise, freq = sf.read('noise.wav', dtype='float32')
        #print(noise.shape)
            print(speechFile.shape)
            noise = noise[:speechFile.shape[0]]   
 
            if no < 15:
                noisePower = np.mean(noise**2)
                speechPower = np.mean(speechFile**2)
                amplification = np.sqrt((speechPower/noisePower)*(10**(-(20/10))))
                speechFile = speechFile+(amplification*noise)
                noisePower1 = np.mean((amplification*noise)**2)
                print(noisePower1)
                # sd.play(speechFile, 16000)
                # sd.wait()
                # sd.play(speechFile, 16000)
                # sd.wait()
    
            mag_frames = toMagFrames(speechFile)
            file_energy = getEnergy(speechFile)
            mfccFile = applyFbankLogDCT(mag_frames, file_energy)
            
            #| Saving mfccs by name and number
            np.save(f'TEST\\{name}_{no}', mfccFile)
            no = no + 1
    
    test_labels = []
    test_data = []
 
    for mfcc_file in sorted(glob.glob(f'TEST\\*.npy')):
        ## load the mfcc data
        mfcc_data = np.load(mfcc_file)
        #print(mfcc_file)
        #| padding the mfcc data to ensure that the mfccs are the same dimensions, using the previously calculated 
        #| max lengths for both dimensions
        mfcc_data = np.pad(mfcc_data, ((0,max_length0-mfcc_data.shape[0]), (0, max_length1-mfcc_data.shape[1]))) 
        #| Appending this to training data
        test_data.append(mfcc_data)
        #print(mfcc_data.shape)
        #| Getting the file name
        stemFileName = (Path(os.path.basename(mfcc_file)).stem)
        #| Splitting the file by _ to get the name rather than number
        label = stemFileName.split('_')
        #print(label)
        #| Appending the name to array of labels to match the classes
        test_labels.append(label[0])
    #print(f'testlabels {test_labels}')
    test_labels = np.array(test_labels)
    #print(test_labels[0])
    test_data = np.array(test_data)
    classes = []
    for data in range(len(test_data)):
        test_data[data] = test_data[data] / np.max(test_data[data])
    
    for val in names.values():
        classes.append(val)
        #print(classes)
    
    
    LE = LabelEncoder()
    #print(classes)
    LE = LE.fit(classes)
    test_labels = to_categorical(LE.transform(test_labels))

    #| Creating model for predictions
    model = createDNN()
    model.load_weights('name_attempt_distortion2sd.weights.h5')   
    predictions = model.predict(test_data, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_class = LE.inverse_transform(predicted_labels)
    #| Trying to get the actual labels following the same process of predicted labels transformation
    actual_labels = np.argmax(test_labels, axis=1)
    for prediction in range(len(predicted_labels)):
        #| Getting the actual label and the predicted class 
        predicted_class = LE.inverse_transform(predicted_labels)
        actual_label = LE.inverse_transform(actual_labels)

        print(f'prediction: {predicted_class[prediction]}, actual: {actual_label[prediction]}')
        
    accuracy = metrics.accuracy_score(actual_labels, predicted_labels)
    print(f'Accuracy: {accuracy * 100}%')
    
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(test_labels,axis=1), predicted_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
                                                confusion_matrix)
    cm_display.plot()
#testing_recordings()