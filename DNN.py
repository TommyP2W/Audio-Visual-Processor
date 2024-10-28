# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:03:02 2024

@author: tommy
"""
import tensorflow as tf
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
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics 
import glob as glob
import soundfile as sf 
import numpy as np
from main import applyFbankLogDCT, toMagFrames
import matplotlib.pyplot as plt
import tensorflow as tf
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


### Loading in all audio files found in directory
for name in names:
    no = 0
    print(name)
    name = names[name]
    for soundfile in sorted(glob.glob(f'AVPSAMPLESMONO\\{name}\\*.wav')):
            speechFile, frequency = sf.read(soundfile)
            #print(speechFile.shape)
            mag_frames = toMagFrames(speechFile)
            mfccFile = applyFbankLogDCT(mag_frames)
            #print(mfccFile.shape)
            #plt.imshow(mfccFile, origin='lower')
            #plt.show()

            np.save(f'mfccs\\{name}\\{name}{no}.py', mfccFile)
            no = no + 1
            #print(soundfile)

speechFile, fs = sf.read('look_out.wav', dtype='float32')
a = toMagFrames(speechFile)
c = applyFbankLogDCT(a)
#.imshow(c, origin='lower')

### TO BE DONE AFTER MFCC CONFIRMED CORRECT

# data = []
# labels = []
# i = 0
# max_length1 = 0
# max_length0 = 0
# for mfcc_file in sorted(glob.glob('mfccs/Joey/Joey2*.py.npy')):
#     mfcc_data = np.load(mfcc_file)
#     if mfcc_data.shape[0] > max_length0:
#         max_length0 = mfcc_data.shape[0] 
#     if mfcc_data.shape[1] > max_length1:
#         max_length1 = mfcc_data.shape[1]
# print(max_length1)
# print(max_length0)
# for mfcc_file in sorted(glob.glob('mfccs/Joey/Joey2*.py.npy')):
#     mfcc_data = np.load(mfcc_file)
#     print(mfcc_data.shape)
#     print(mfcc_file)
    

#     mfcc_data = np.pad(mfcc_data, ((0,max_length0-mfcc_data.shape[0]), (0, max_length1-mfcc_data.shape[1]))) 
#     data.append(mfcc_data)
    
#     stemFileName = (Path(os.path.basename(mfcc_file)).stem)
#     label = stemFileName.split('_')
#     labels.append(label[0])

# data = np.array(data)
# labels = np.array(labels)


# data = data / np.max(data)

# LE = LabelEncoder()
# classes = ['Joey', 'Noah']
# LE = LE.fit(classes)
# labels = to_categorical(LE.transform(labels))                
# X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
# X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

# def createDNN():
#     numClasses = 10
#     model = Sequential()
#     model.add(InputLayer(input_shape=(40,21,1)))
#     model.add(Conv2D(128,(3,3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3,3)))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(numClasses))
#     model.add(Activation('softmax'))
#     return model
    
# num_epochs = [80]
# num_batch_size = [80]
# model = createDNN()
# model.compile(loss='categorical_crossentropy',
# metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
# model.summary()

# for i in range(len(num_epochs)):
#     history = model.fit(X_train, y_train, validation_data=(X_val,
#     y_val), batch_size=num_batch_size[i], epochs=num_epochs[i],
#     verbose=1)

#     model.save_weights('digit_classification.weights.h5')

#     model = createDNN()
#     model.compile(loss='categorical_crossentropy',
#     metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
#     #model.load_weights('digit_classification.weights.h5')

#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()

# predicted_probs=model.predict(X_test,verbose=0)
# predicted=np.argmax(predicted_probs,axis=1)
# actual=np.argmax(y_test,axis=1)
# accuracy = metrics.accuracy_score(actual, predicted)
# print(f'Accuracy: {accuracy * 100}%')

# predicted_prob = model.predict(np.expand_dims(X_test[0,:,:],
# axis = 0), verbose=0)
# predicted_id = np.argmax(predicted_prob,axis=1)
# predicted_class = LE.inverse_transform(predicted_id)

# confusion_matrix = metrics.confusion_matrix(
# np.argmax(y_test,axis=1), predicted)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
# confusion_matrix)
# cm_display.plot()




















            


            
            
# # mfcc_file = np.load('mfccs/Joey/Joey20.py.npy')
# # mfcc_file2 = np.load('mfccs/Joey/Joey1.py.npy')


# # plt.imshow(c.T[:50, :10], origin='lower')
# # # plt.imshow(mfcc_file2.T[40:], origin='lower')
# # # plt.show()


# # ### Suck my bolloc
# # #def feedNeuralNetwork():
    