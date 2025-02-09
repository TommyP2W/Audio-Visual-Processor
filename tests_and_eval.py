from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
from pathlib import Path
import glob
import cv2 
from sklearn import metrics
from feature_processing import names, audio_npy, visual_npy, get_max_dimensionality
from main import toMagFrames, getEnergy, applyFbankLogDCT
from visual_preprocessing import dct8by8, detect_face, detect_mouth
from Model import createDNN, train_test_split_method, li_test_integ


def test_voice(dimensions):
    """
         This function takes the microphone as an input, the function will record the user's voice for two seconds.
         Once recording has been completed, the voice data is processed and against a previously trained model is
         classified into the groups of names specified in feature_processing.py.

         Args:
            dimensions: Dimensions for the audio data.
         Returns:
            Returns Nothing.
     """

    model = createDNN((dimensions[0], dimensions[1], 1))
    model.load_weights('new.weights.h5')   
   
    voice_test = []
    speech_file = sd.rec(2 * 16000, samplerate=16000, channels=1, dtype='float32')
    
    print(speech_file.shape)
 
    print("recording")
    sd.wait()
    # sd.play(speechFile, 16000)
    print("Done!")
    classes = []
    for val in names.values():
        classes.append(val)

    # Normalising each data index by its own highest value, this is to account for differing amplitudes per mfcc

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(classes)

    speech_file = 0.99 * speech_file / max(abs(speech_file))
    speech_file = speech_file.squeeze()

    mag_frames = toMagFrames(speech_file)
    file_energy = getEnergy(speech_file)
    mfcc_data = applyFbankLogDCT(mag_frames, file_energy)

    plt.imshow(mfcc_data, origin='lower')
    plt.show()
    
    mfcc_data = np.pad(mfcc_data, ((0,dimensions[0]-mfcc_data.shape[0]), (0, dimensions[1]-mfcc_data.shape[1])))
    plt.imshow(mfcc_data, origin='lower')
    voice_test.append(mfcc_data)

    voice_test = np.array(voice_test)

    for val in range(len(voice_test)):
        voice_test[val] = voice_test[val] / np.max(voice_test[val])

    predicted_name = model.predict(voice_test, verbose=0)
    predicted_id = np.argmax(predicted_name, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_id)

    print(f'Predicted name {predicted_class}')


def quick_record():
    """
         This function was built to quickly record name samples. Particularly for when you need to produce lots of
         samples to build the network. Records the user's voice for two seconds, then saves into directory.

         Args:

         Returns:
             Returns Nothing.
     """
    # For every name in dictionary names
    for name in names:
        # Getting the actual name value by index, would otherwise be a number for the index of names
        name = names[name] 
        # Making new child directory for each name
        os.mkdir(f'AVPSAMPLESSD\\{name}')
        # Used for making a max of 29 samples per name
        index = 0     
        while index < 30:
            # Debugging to let me know it was ready to say name for the ith time
            print(f'recording {name} {index}')   
            speech_file = sd.rec(2* 16000, samplerate=16000, channels=1, dtype='float32')
            sd.wait()
            print("done")
            speech_file = 0.99 * speech_file / max(abs(speech_file))
            # conditional is primarily used to keep consistency with other samples
            if index < 10:
                sf.write(f'AVPSAMPLESSD\\{name}\\{name}00{index}sd.wav', speech_file, 16000)
            else:
                sf.write(f'AVPSAMPLESSD\\{name}\\{name}0{index}sd.wav', speech_file, 16000)
            index += 1


def testing_recordings(dimensions):
    """
            This function tests the models ability to adapt to more realistic environments. The testing data is
            injected with artificial noise and predicts the names based of the augmented data.

            Args:

            Returns:
                Returns Nothing.
        """

    """ This section grabs each audio file, adds noise and saves as a npy file for testing data"""
    for name in names:
        name = names[name] 
        index = 0
        for file in sorted(glob.glob(f'AVPSAMPLESSD\\TEST\\{name}\\*.wav')):

            speech_file, frequency = sf.read(file, dtype='float32')
            noise, freq = sf.read('noise.wav', dtype='float32')
            noise = noise[:speech_file.shape[0]]
 
            if index < 15:
                noise_power = np.mean(noise**2)
                speech_power = np.mean(speech_file**2)
                amplification = np.sqrt((speech_power/noise_power)*(10**(-(20/10))))
                speech_file = speech_file+(amplification*noise)
                noise_power1 = np.mean((amplification*noise)**2)
    
            mag_frames = toMagFrames(speech_file)
            file_energy = getEnergy(speech_file)
            mfcc_file = applyFbankLogDCT(mag_frames, file_energy)
            
            # Saving mfccs by name and number
            np.save(f'TEST\\{name}_{index}', mfcc_file)
            index += 1
    
    test_labels = []
    test_data = []
 
    # This section appends the data and labels to individual arrays
    for mfcc_file in sorted(glob.glob(f'TEST\\*.npy')):
        mfcc_data = np.load(mfcc_file)
    
        mfcc_data = np.pad(mfcc_data, ((0, dimensions[0]-mfcc_data.shape[0]), (0, dimensions[1]-mfcc_data.shape[1])))
        test_data.append(mfcc_data)
    
        stem_file_name = Path(os.path.basename(mfcc_file)).stem
        label = stem_file_name.split('_')
      
        test_labels.append(label[0])

    test_labels = np.array(test_labels)
    test_data = np.array(test_data)
    classes = []

    # Normalising and adding classes to Label encoder to be transformed
    for data in range(len(test_data)):
        test_data[data] = test_data[data] / np.max(test_data[data])
    
    for val in names.values():
        classes.append(val)
        # print(classes)
    
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(classes)
    test_labels = to_categorical(label_encoder.transform(test_labels))

    # Loading in weights and creating DNN
    model = createDNN((dimensions[0], dimensions[1]))
    model.load_weights('name_attempt_distortion2sd.weights.h5')   

    predictions = model.predict(test_data, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_labels)
    actual_labels = np.argmax(test_labels, axis=1)

    # for loop to get all the predictions and actual labels printed to console
    for prediction in range(len(predicted_labels)):
        # Getting the actual label and the predicted class
        predicted_class = label_encoder.inverse_transform(predicted_labels)
        actual_label = label_encoder.inverse_transform(actual_labels)

        print(f'prediction: {predicted_class[prediction]}, actual: {actual_label[prediction]}')
        
    accuracy = metrics.accuracy_score(actual_labels, predicted_labels)
    print(f'Accuracy: {accuracy * 100}%')
    
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(test_labels, axis=1), predicted_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
                                                confusion_matrix)
    cm_display.plot()


def test_visual_example(video, dimensions):
    """
            This function was built to test individual video samples. It grabs the video requested, processes the data
            and then asks the network to predict the name said in the video based off of facial landmarks.

            Args:

            Returns:
                Returns Nothing.
    """

    model = createDNN((dimensions[0], dimensions[1], dimensions[2]))
    model.load_weights('visual.weights.h5')
    cap = cv2.VideoCapture(video)
    frames_arr = []

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            print("frame detected")
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_arr.append(grey)
        else:
            break

    testing_data = []
    mfcc_data = []
    classes = []

    """Grabbing the coefficients for each frame of video"""
    for frame in frames_arr:
        faces = detect_face(frame)
        detected_mouth = detect_mouth(faces)
        if detected_mouth is None:
            break
        print(detected_mouth)
        dct_8x8 = dct8by8(detected_mouth)
        mfcc_data.append(dct_8x8)

    mfcc_data = np.array(mfcc_data)

    if mfcc_data.shape[0] > dimensions[0]:
        dimensions[0] = mfcc_data.shape[0]
        
    for val in names.values():
        classes.append(val)

    vid_coefficients = np.pad(mfcc_data, ((0, dimensions[0]-mfcc_data.shape[0]),
                                          (0, dimensions[1]-mfcc_data.shape[1]),
                                          (0, dimensions[2]-mfcc_data.shape[2])))
    
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(classes)
    
    testing_data.append(vid_coefficients)
    testing_data = np.array(testing_data)

    predicted_name = model.predict(testing_data, verbose=0)
    predicted_id = np.argmax(predicted_name, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_id)

    print(f'Predicted name {predicted_class}')


"""This code was to optimise the CNN without importing GridSearch, redundant."""
# def optimise():
#     param_grid = {
#         'epochs': [10, 15,20],
#     }
#     #print(X_train.shape)
#     a = createDNN((max_visual0,max_visual1,max_visual2))
#     a.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

#     model = KerasClassifier(model=a)
#     print(model.get_params().keys())
#     gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
#     gs = gs.fit(vis_x_train, vis_y_train)
#     ga = gs.cv_results_
#     print(ga)
#     #gs = gs.predict(vis_x_test)
#     print(f'Best params: {gs.best_params_}')
#     print(f'Best score: {gs.best_score_}')
# #optimise()

"""Function to test late integration"""


def avp_li(audio_dimensions, visual_dimensions,
          visual_data, visual_labels, audio_data, audio_labels):

    """
            This function was built to test late integration of the visual and audio networks. Two CNNs are created,
            two sets of data (audio, visual) are then split into training and testing sets. The predictions are combined
            using argmax() to determine the final prediction made by both networks.

            Args:

            Returns:
                Returns Nothing.
    """

    visual_network = createDNN((visual_dimensions[0],
                                visual_dimensions[1],
                                visual_dimensions[2]))
    audio_network = createDNN(((audio_dimensions[0],
                                audio_dimensions[1]
                                , 1)))
    
    visual_network.load_weights('bitmap.weights.h5')
    audio_network.load_weights('new_aud.weights.h5')

    (vis_x_train, vis_y_train, vis_x_test, vis_y_test,
     vis_x_val, vis_y_val, vis_le) = train_test_split_method(visual_data, visual_labels)

    (aud_x_train, aud_y_train, aud_x_test, aud_y_test,
     aud_x_val, aud_y_val, aud_le) = train_test_split_method(audio_data, audio_labels)

    aud_x_test, vis_x_test, aud_y_test, vis_y_test = li_test_integ(vis_y_test, vis_x_test,
                                                                   aud_y_test, aud_x_test) 
    classes = []
    for val in names.values():
        classes.append(val)

    # print(aud_x_test.shape.shape)
    # print(f'visual-shape: {vis_x_test.shape}')

    # Getting the predictions from the visual network
    vis_predictions = visual_network.predict(vis_x_test, verbose=0)
    vis_predicted_labels = np.argmax(vis_predictions, axis=1)
    vis_predicted_class = vis_le.inverse_transform(vis_predicted_labels)
    vis_true_labels = np.argmax(vis_y_test, axis=1)
    vis_true_labels = vis_le.inverse_transform(vis_true_labels)

    print(vis_true_labels)
    print(f'{vis_predicted_class.shape} \n')
    vis_accuracy = metrics.accuracy_score(vis_true_labels, vis_predicted_class)
    print(vis_true_labels)
    print(vis_predicted_class)
    print(f'Visual Accuracy: {vis_accuracy * 100}%')
    print(vis_true_labels.shape)
     
    # Getting the predictions from the audio network
    aud_predictions = audio_network.predict(aud_x_test, verbose=0)
    aud_predicted_labels = np.argmax(aud_predictions, axis=1)
    aud_predicted_class = aud_le.inverse_transform(aud_predicted_labels)
    aud_true_labels = np.argmax(aud_y_test, axis=1)
    aud_true_labels = aud_le.inverse_transform(aud_true_labels)
    aud_accuracy = metrics.accuracy_score(aud_true_labels, aud_predicted_class)
    print(f'Audio network: {aud_accuracy * 100}%')

    # print(f'{aud_true_labels} \n')

    # Combining and getting the maximum value for predictions
    print(aud_predictions.shape)
    print(f'visual-shape: {vis_predictions.shape}')
    combined = []
    for row in range(len(aud_predictions)):
        combined_pred = aud_predictions[row] + vis_predictions[row]
        combined.append(np.argmax(combined_pred))

    combined_labels = aud_le.inverse_transform(combined)

    accuracy = metrics.accuracy_score(vis_true_labels, combined_labels)
    print(f'LI Accuracy: {accuracy * 100}%')

    # Confusion matrix for Late Integration, Visual network and Audio Network
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(aud_y_test, axis=1), aud_predicted_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Confusion Matrix - Audio Network')
    plt.show()

    confusion_matrix = metrics.confusion_matrix(
        np.argmax(vis_y_test, axis=1), vis_predicted_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Confusion Matrix - Visual Network')
    plt.show()

    confusion_matrix = metrics.confusion_matrix(
        np.argmax(vis_y_test, axis=1), combined)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.title('Confusion Matrix - Late Integration')

    plt.show()


vid = r'C:\Users\tommy\Desktop\AudiovisualLabs\AVPSum\VisualClips\Emilija005.mp4'
visual_dimensions = get_max_dimensionality('visual')
audio_dimensions = get_max_dimensionality('audio')
visual_arr, visual_labels = visual_npy(visual_dimensions)
aud_arr, aud_labels = audio_npy(audio_dimensions)
#avp_li(audio_dimensions, visual_dimensions, visual_arr, visual_labels, aud_arr, aud_labels)

test_voice(audio_dimensions)
