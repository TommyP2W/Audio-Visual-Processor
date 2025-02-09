from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from feature_processing import names


def train_test_split_method(data, labels):

    """
    Helper function for creating training and testing splits.

    Args:
        data: Contains npy file data.
        labels: Contains labels for processed npy files.

    Returns:
        data_x_train, data_y_train: Contains training data for fitting the network.
        data_x_test, data_y_test: Contains testing data for making predictions.
        data_x_val, data_y_val: Contains validation data.
        data_LE: Label encoder.
    """

    classes = []

    # Appending all values in defined name dictionary to a classes array
    for val in names.values():
        classes.append(val)


    combined_LE = LabelEncoder()
    combined_LE = combined_LE.fit(classes)
    categorical_labels = to_categorical(combined_LE.transform(labels))
   

    com_x_train, com_x_tmp, com_y_train, com_y_tmp = train_test_split(data, categorical_labels, test_size=0.2,
                                                                    random_state=0, stratify=categorical_labels,
                                                                    shuffle=True)
    com_x_val, com_x_test, com_y_val, com_y_test = train_test_split(com_x_tmp, com_y_tmp, test_size=0.5, random_state=0,
                                                                    stratify=com_y_tmp,
                                                                    shuffle=True)
   
    return com_x_train, com_y_train, com_x_test, com_y_test ,com_x_val, com_y_val, combined_LE


def li_test_integ(vis_y_test, vis_x_test, aud_y_test, aud_x_test):
    """
    Helper function to align audio and visual data for late integration testing.

    Args:
        vis_y_test, vis_x_test: Contains testing data for visual features.
        aud_y_test, aud_x_test: Contains testing data for audio features.

    Returns:
        aud_test_data, vis_test_data: New numpy array with matching data.
        aud_test_labels, vis_test_labels: New numpy array with matching labels.
    """

    aud_test_labels = []
    aud_test_data = []
    vis_test_data = []
    vis_test_labels = []

    for row in range(len(vis_y_test)):
        for col in range(len(aud_y_test)):
            if np.array_equal(vis_y_test[row], aud_y_test[col]):
                aud_test_data.append(aud_x_test[col])
                vis_test_data.append(vis_x_test[row])
                aud_test_labels.append(aud_y_test[col])
                vis_test_labels.append(vis_y_test[row])

    aud_test_data = np.array(aud_test_data)
    vis_test_data = np.array(vis_test_data)
    aud_test_labels = np.array(aud_test_labels)
    vis_test_data = np.array(vis_test_data)

    return aud_test_data, vis_test_data, aud_test_labels, vis_test_labels

def createDNN(input_params):

    """
    Helper function for creating CNN network.

    Args:
        input_params: Tuple containing shape for input layer.

    Returns:
        model: Created network. 
    """


    numClasses = 20
    model = Sequential()
    model.add(InputLayer(shape=(input_params)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))

    return model



def train_model(option_critic,dimensions, data, labels):

    """
    Function for compiling and training model. Provides plot after completion detailing loss and accuracy across epoch.
    Args:
        option_critic: String containing option for the type of visual feature.
        dimensions: Array containing integers to determine input layer size.
        data: Array containing normalised and padded input data.
        labels: Array containing labels for input data.

    Returns:
        This function returns nothing.
    """
    
    num_epochs = 25
    num_batch_size = 32
   
    if option_critic == 'visual':
        model = createDNN((dimensions[0], 
                           dimensions[1],
                           dimensions[2]))
    else: 
        model = createDNN((dimensions[0], dimensions[1]))
    
    data_x_train, data_y_train, data_x_test, data_y_test, data_x_val, data_y_val = train_test_split(data, labels)

    # Slower learning rate, reduced by factors of 10, to account for more layers and to reduce intial model loss
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
   
    
    model.summary()

    # Fitting the model with training and validation data
    training = model.fit(data_x_train, data_y_train, validation_data=(data_x_val,
    data_y_val), batch_size=num_batch_size, epochs=num_epochs,
    verbose=1)
   
    # Saving the weights to remove the need to completely retrain the model
    model.save_weights('bitmap.weights.h5')

    # Plotting the loss and the accuracy
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
    
def prediction(option_critic, dimensions, data_x_test, data_y_test, combined_LE):

    """
    Function to make predictions on a given testing set. Produces confusion matrix to provide visual representation of performance.
    Args:
        option_critic: String containing option for the type of visual feature.
        dimensions: Array containing integers to determine input layer size.
        data_x_test: Array containing normalised and padded testing data.
        data_y_test: Array containing labels for input testing data.

    Returns:
        This function returns nothing.
    """

    if option_critic == 'visual':
        model = createDNN((dimensions[0], 
                           dimensions[1],
                           dimensions[2]))
    else: 
        model = createDNN((dimensions[0], dimensions[1]))
    
    model.load_weights('visual.weights.h5')

    predicted_probs=model.predict(data_x_test, verbose=0)
    predicted=np.argmax(predicted_probs,axis=1)

    actual=np.argmax(data_y_test,axis=1)
    actual_class = combined_LE.inverse_transform(actual)
    predicted_class = combined_LE.inverse_transform(predicted)
    accuracy = metrics.accuracy_score(actual, predicted)
    
    print(f'Accuracy: {accuracy * 100}%')

    for name in range(len(predicted_class)):
        print(f'Actual class: {actual_class[name]}, Predicted class: {predicted_class[name]}')

    #| Plotting confusion matrix
    confusion_matrix = metrics.confusion_matrix(
        np.argmax(data_y_test,axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =
                                                confusion_matrix)
    cm_display.plot()
    plt.show()
#prediction()
#| This function was created to take live audio recordings and get a prediction based off
#| the current model weightings
