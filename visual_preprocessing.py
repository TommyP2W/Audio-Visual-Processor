import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci_img
from scipy.fftpack import dct, dctn, idctn


# def load_image():
#     img = cv2.imread(r"C:\Users\tommy\Pictures\Camera Roll\b.png")
#     print(img)
#     gray_img = img[:, :, 0]
#     # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # plt.imshow(gray_img)
#     # plt.show()
#     return gray_img


def detect_face(gray_img):
    """
    Function to detect the face from gray_scaled image. Using cs2.CascadeClassifier, the function extracts
    x,y,w,h (x,y representing the top left coords of the image, w,h representing the width and height of the image).
    These combined draws a box around the face which can then be used further.

    Args:
        gray_img: image of the face.
    Returns:
        face_detected: lower half of the face to focus the mouth.
    """

    # Initialise face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +
                                            "haarcascade_frontalface_default.xml")

    # Detect the face from the gray_img
    face = face_classifier.detectMultiScale(gray_img, scaleFactor=1.15,
                                        minNeighbors=12, minSize=(20, 20))
    # print(face)
    if len(face) == 0:
        print("No face detected")
        plt.imshow(gray_img)
        plt.show()
        return None
    # Get bounding coords
    x = face[0][0]
    y = face[0][1]
    w = face[0][2]
    h = face[0][3]

    # gray_img = np.array(gray_img)
    # for x, y, w, h in face:
    #     cv2.rectangle(gray_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.imshow("Face", gray_img)
    # cv2.waitKey(1000)
    # if cv2.waitKey(1) & 0xFF == ord('q'):

    # Draw the face detected
    face_detected = gray_img[int(y + h / 2):y + h, x:x + w]  # Correct slicing

    return face_detected


def detect_mouth(face_detected):
    """
    Function to detect the mouth from the lower half of the face. Using cs2.CascadeClassifier, the function extracts
    x,y,w,h (x,y representing the top left coords of the image, w,h representing the width and height of the image).
    These combined draws a box around the mouth which can then be used further.

    Args:
       face_detected: face detected by the detected mouth
    Returns:
       mouth_detected: mouth detected.
    """

    # Initialise the mouth classifier
    mouth_classifier = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    # Detect the mouth from the classifier
    mouth = mouth_classifier.detectMultiScale(face_detected, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    # Get the mouth coords

    if len(mouth) == 0:
        return None
    # print(mouth)
    mx = mouth[0][0]
    my = mouth[0][1]
    mw = mouth[0][2]
    mh = mouth[0][3]

    # Draw the mouth
    # face_detected = np.array(face_detected)
    # for x, y, w, h in mouth:
    #     cv2.rectangle(face_detected, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.imshow("Face", face_detected)
    # cv2.waitKey(1000)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     pass

    mouth_detected = face_detected[my:my + mh, mx:mx + mw]
    return mouth_detected


def get_frame(video):
    """
    Function to get individual frames from a video file. Loads in the video file, then reads and processes every
    frame detected.

    Args:
        video: video file.
    Returns:
        frames_arr: Array of frames.
    """

    # Loading in a testing sample
    cap = cv2.VideoCapture(video[1])
    frames_arr = []
    # While the capture object still has frames to process
    while cap.isOpened():
        # Process a frame
        ret, frame = cap.read()
        if frame is not None:
            # print("frame detected")
            # Convert to grayscale
            gray_img = frame[:, :, 0]
            # plt.imshow(gray_img, cmap='gray')
            # plt.show()
            frames_arr.append(gray_img)
        else:
            break

    return frames_arr


def dct8by8(detected_mouth):
    """
     Function to get the dct coefficients from 8x8 blocks of an image. This function applies the discrete
     courier transformation to each 8x8 block, then returns these coefficients. The purpose of using 8x8 blocks
     is to isolate a small number of coefficients, so that it can remove the high frequencies from that block. If
     you were to dct an entire image, this may cause high frequencies with meaning to get filtered out, as one
     low frequency can cause bias.

    Args:
        detected_mouth: mouth detected.
    Returns:
        dct_array: returns array of dct coefficients.
    """

    # Resize the mouth, this helps with making sure the loop conditions are consistent as sizes may vary
    if len(detected_mouth) == 0:
        return None
    resized_mouth = cv2.resize(detected_mouth, (64, 64))

    # Create a dct array for the dimensions of the resized mouth
    dct_arr = np.zeros((resized_mouth.shape[0], resized_mouth.shape[1]))
    # Creating an array for each individual block, this is used to confirm the blocking process is correct.
    block_arr = np.zeros((resized_mouth.shape[0], resized_mouth.shape[1]))
    # print(car.shape)

    # for row in range of the height of the image, stepping 8 times
    for row in range(0, resized_mouth.shape[0], 8):
        # for row in range of the width, stepping 8 times
        for col in range(0, resized_mouth.shape[1], 8):

            # at block index row to row + 8 steps, save the resized mouth pixels at that index
            block_arr[row:row+8,col:col+8] = resized_mouth[row:row+8, col:col+8]

            # dct the image, the image is two-dimensional  so needs to be done twice, not sure how yet.
            dct_temp = dctn(block_arr[row:row+8, col:col+8], type=2, norm="ortho")
            # dct_temp = idctn(dct_temp, type=2, norm="ortho")

            # dct_temp = idct(dct_temp)

            # Save the dct into the dct array
            dct_arr[row:row+8, col:col+8] = dct_temp

    # plt.imshow(dct_arr)
    # plt.show()
    plt.imshow(dct_arr)
    plt.show()
    print(dct_arr.shape)
    return dct_arr


def bitmap_smooth(detected_mouth):
    _, thresh = cv2.threshold(detected_mouth, 130, 255, cv2.THRESH_BINARY)
    thresh = ~thresh
    label_structure = [[0,1,0],
                       [1,1,1],
                       [0,1,0]]
    labeled_img, num_features = sci_img.label(thresh, label_structure)
    small_removed_img = np.zeros_like(thresh)
    for index in range(1, num_features + 1):
        component_mask = (labeled_img == index)
        if np.sum(component_mask) >= 50:
            small_removed_img[component_mask] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    smoothed_img = cv2.morphologyEx(small_removed_img, cv2.MORPH_CLOSE, kernel)
    smoothed_img = cv2.resize(smoothed_img, (32, 32))
    # print(smoothed_img.shape)
    # plt.imshow(smoothed_img)
    # plt.show()
    return smoothed_img


image = load_image()
faces = detect_face(image)
detected_mouth = detect_mouth(faces)

dct8by8(detected_mouth)
bitmap = bitmap_smooth(detected_mouth)
plt.imshow(bitmap)
plt.show()

kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(detected_mouth, kernel, iterations=2)
_, thresh = cv2.threshold(detected_mouth, 85, 255, cv2.THRESH_TOZERO)
area_of_lips = np.sum(thresh == 0)


