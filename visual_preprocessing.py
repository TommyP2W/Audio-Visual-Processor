
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sci_img
from scipy.fftpack import dct, dctn, idctn


def load_image():
    img = cv2.imread(r"C:\Users\tommy\Pictures\Camera Roll\Screenshot 2024-12-01 154350.png")
    print(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def detect_face(gray_img):
    """
      #| Description:
      #|
      #| Function to detect the face from gray_scaled image. Using cs2.CascadeClassifier, the function extracts
      #| x,y,w,h (x,y representing the top left coords of the image, w,h representing the width and height of the image).
      #| These combined draws a box around the face which can then be used further.
      #|
      #| Parameters:
      #|
      #| gray_img: image of the face.
      #| Return:
      #
      #| face_detected: lower half of the face to focus the mouth.
      #|
      """
    # Initialise face classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +
                                            "haarcascade_frontalface_default.xml")

    # Detect the face from the gray_img
    face = face_classifier.detectMultiScale(gray_img, scaleFactor=1.15,
                                        minNeighbors=5, minSize=(40, 40))
    # Get bounding coords
    x = face[0,0]
    y = face[0,1]
    w = face[0,2]
    h = face[0,3]
    # Draw the face detected
    face_detected = gray_img[int((y + h) / 2):y + h, x:x + w]  # Correct slicing

    plt.imshow(face_detected, cmap='gray')
    plt.show()
    return face_detected
# print(gray_img.shape)
# x = gray_img.flatten()
# b = sum(x) / len(x)
# c = gray_img - b
# print(c)


def detect_mouth(face_detected):
    """
    #| Description:
    #|
    #| Function to detect the mouth from the lower half of the face. Using cs2.CascadeClassifier, the function extracts
    #| x,y,w,h (x,y representing the top left coords of the image, w,h representing the width and height of the image).
    #| These combined draws a box around the mouth which can then be used further.
    #|
    #| Parameters:
    #|
    #| face_detected: face detected by the detected mouth
    #| Return:
    #|
    #| mouth_detected: mouth detected.
    #|
    """
    # Initialise the mouth classifier
    mouth_classifier = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
    # Detect the mouth from the classifier
    mouth = mouth_classifier.detectMultiScale(face_detected, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    # Get the mouth coords
    mx = mouth[0, 0]
    my = mouth[0, 1]
    mw = mouth[0, 2]
    mh = mouth[0, 3]
    # Draw the mouth
    mouth_detected = face_detected[my:my + mh, mx:mx + mw]
    return mouth_detected


def get_frame():
    """
      #| Description:
      #|
      #| Function to get individual frames from a video file. Loads in the video file, then reads and processes every
      #| frame detected.
      """
    # Loading in a testing sample
    cap = cv2.VideoCapture(r"C:\Users\tommy\Pictures\Camera Roll\VisualClips\Amelia\Amelia014.mp4")

    print(cap)
    # While the capture object still has frames to process
    while cap.isOpened():
        # Process a frame
        ret, frame = cap.read()
        # Convert to grayscale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grey = frame[:,:,2]
        # grey = grey[:,:,2]
        plt.imshow(grey)
        plt.show()


def dct8by8(detected_mouth):
    """
      #| Description:
      #|
      #| Function to get the dct coefficients from 8x8 blocks of an image. This function applies the discrete
      #| courier transformation to each 8x8 block, then returns these coefficients. The purpose of using 8x8 blocks
      #| is to isolate a small number of coefficients, so that it can remove the high frequencies from that block. If
      #| you were to dct an entire image, this may cause high frequencies with meaning to get filtered out, as one
      #| low frequency can cause bias.
      #|
      #| Parameters:
      #|
      #| mouth_detected: mouth detected.
      #| Return:
      #|
      #| dct_array: returns array of dct coefficients.
      #|
      """
    # Resize the mouth, this helps with making sure the loop conditions are consistent as sizes may vary
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
            # print(block.shape)
            # dct the image, the image is two-dimensional  so needs to be done twice, not sure how yet.
            dct_temp = dctn(block_arr[row:row+8, col:col+8], type=2, norm="ortho")
            dct_temp = idctn(dct_temp, type=2, norm="ortho")


            # dct_temp = idct(dct_temp)

            # Save the dct into the dct array
            dct_arr[row:row+8, col:col+8] = dct_temp

            print(dct_arr[0][0])
    plt.imshow(dct_arr)
    plt.show()
    return dct_arr

# blockydct = dct8by8()
# print(blockydct)
# plt.imshow(cvd, origin='lower')
# plt.show()


imag = load_image()
faces = detect_face(imag)
detected_mouth = detect_mouth(imag)
plt.imshow(detected_mouth)
plt.show()
plt.imshow(faces)
plt.show()
dct8by8(detected_mouth)

kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(detected_mouth, kernel, iterations=2)
_, thresh = cv2.threshold(detected_mouth, 85, 255, cv2.THRESH_TOZERO)
area_of_lips = np.sum(thresh == 0)

# print(area_of_lips)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
a = cv2.drawContours(thresh, contours, -1, (255, 0, 0), 1)

gaussian_filter = sci_img.gaussian_laplace(detected_mouth, 1)
plt.imshow(thresh)
# plt.imshow(dilated)

plt.show()
