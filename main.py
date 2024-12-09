# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:34:53 2024

@author: scott
"""


import numpy as np
import scipy as spy
import plotly.io as pio
import plotly.express as px
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile as sf 
import math

# Input: 320 freq speech frame. Output: The mag spec of the inputted speech
def extractMag (speechFrame):
    hamSpeech = np.squeeze(speechFrame) * np.hamming(320) #320 hard coded as we know all audio is recorded at 16kHz- 320Hz is 20ms
    speechDFT = np.fft.fft(hamSpeech) #hamming windowed + put into dft calc
    mag = np.abs(speechDFT)
    mag = mag[0:int(len(mag)/2)] #removing the reflected part of the mag spec
    return mag

# Input: speech file of any length at 16kHz. Output: List of mag specs, of the frames
def toMagFrames(speechFile):
    num_samples = len(speechFile)
    frame_length = 320 #this is 20ms as the length of speech file is in freq
    num_of_frames = math.floor((num_samples-frame_length)/(frame_length/2)) #calcs the number of frames, so i can create a array of appropraite size
    mag_frames = np.zeros(num_of_frames+1) #empty list that can fit the number of frames
    mag_frames = [None] * (num_of_frames+1)
    for index in np.arange(0,num_samples-frame_length,int(frame_length/2)):
        startIndex = index # compute first sample of frame
        endIndex = index + frame_length # compute last sample of frame
        shortTimeFrame = speechFile[startIndex:endIndex] # shortTimeFrame = speechFile[firstSample:lastSample]
        mag_frames[int(index/160)] = extractMag(shortTimeFrame) #putting extracted mag into an array of magspec frames
    return mag_frames

# Input: The min freq (normally 0), the Max freq (Our .wavs are all 16kHz), the number of fbanks wanted.
# Output: List of points based on the frequency/frame length. The frames between each set of points are the bins.
def melSpacedPoints (minFreq, maxFreq, fbanks):
    fMelPointsList = np.zeros(fbanks + 2)
    binList = np.zeros(fbanks + 2)
    minfMel = 1125*math.log(1 + minFreq/700)
    maxfMel = 1125*math.log(1 + maxFreq/700)
    freqPerBin = maxFreq/160 #bins are gonna be 160 (so they are the same length as our mag specs (320/2))
    fMelbanks = maxfMel/(fbanks + 2)
    for index in range (0, fbanks + 2):
        melPoint = fMelbanks * (index)
        freqM = 700*(math.exp(melPoint/1125)-1)
        fMelPointsList[index] = freqM
        binList[index] = math.floor(freqM/freqPerBin)
    return binList

#Input: a list of 'bins' from the MelSpacedPoints function (just a list of numbers)
#Output: a list of triangular fbank vectors (which in themselves are lists of points)
def triangularFBank (binlist):
    triVector = np.zeros(160)
    all_fbank_vectors = np.zeros(shape=(len(binlist),160))
    for index in range (1,len(binlist)-1):
        currentPeak = binlist[index]
        pastPeak = binlist[index - 1]
        nextPeak = binlist[index + 1]
        for k in range (0,len(triVector)-1):
            if (k < currentPeak):
                triValue = 0
            if (pastPeak <= k) and (k <= currentPeak):
                triValue = (k - pastPeak)/(currentPeak - pastPeak)
            if (currentPeak <= k) and (k <= nextPeak):
                triValue = (nextPeak - k)/(nextPeak - currentPeak)
            if (k > nextPeak):
                triValue = 0
            triVector[k] = triValue
        all_fbank_vectors[index] = triVector 
        #plt.plot(all_fbank_vectors[index])
    return all_fbank_vectors

# def applyFbankLogDCT(mag_frames):
#     mfcc = [None] * len(mag_frames)
#     for index in range (len(mag_frames)):
#         for k in range (len(tri_fbanks)):
#             #matmul every mag frame but each tribank filter
#             #then log it, then apply dct equation
#             mfcc[index] = np.fft.dct(math.log10(np.matmul(mag_frames[index],tri_fbanks[k])))
#             #removing latter half to try and remove the pitch, may need to change what % is being removed
#             mfcc[index] = mfcc[index[0:int(len(mfcc[index]/2))]]
#     np.save('test', mfcc)
    
def applyFbankLogDCT(mag_frames, file_energy):
    tri_cut = len(tri_fbanks) #length of the fbanks array -2 to take off the first value (0) and the last
    mfcc = np.zeros(shape = ((tri_cut),len(mag_frames))) #with 8 fbanks 8+6+4
    for indexfbank in range (1,len(tri_fbanks)-1):
        for indexmag in range (0, len(mag_frames)-1): #to leave out the first value (0) and the last one too
            #print("mag frames:", len(mag_frames))
            #print("shape of c:", np.shape(c))
            #matmul every mag frame but each tribank filter
            #then log it, then apply dct equation
            #a = np.matmul(mag_frames[indexmag],tri_fbanks[indexfbank])
            #print("a fine:", a)
            #b = math.log10(a)
            #print("b fine:", b)
            #c[indexfbank][indexmag] = b
            matmulled = np.matmul(mag_frames[indexmag],tri_fbanks[indexfbank])
            if (matmulled == 0):
                matmulled = 0.000001 #if you log 0 you get -infinity which will break the dnn
            mfcc[indexfbank-1][indexmag] = math.log10(matmulled)
            #dct is meant to be done via column
            #mfcc[indexfbank-1] = spy.fft.dct(mfcc[indexfbank-1])
            
        # plt.figure(figsize=(10,8))
        # plt.imshow(mfcc, origin='lower')
        # plt.colorbar()
        # plt.show()
    for indexdct in range (0,mfcc.shape[1]-1):   
        
        dct_frame = spy.fft.dct(mfcc[:,indexdct])
        mfcc[:,indexdct] = dct_frame
        #print(len(mfcc[:,indexmag]))
        # plt.figure(figsize=(10,8))
        # plt.imshow(mfcc, origin='lower')
        # # plt.colorbar()
        # plt.show()
        
    # for indexpitch in range (math.floor((2/3) * (mfcc.shape[0]-1)),mfcc.shape[0]): #takes off last 33% of dct- pitch
    #     mfcc[indexpitch] = 0
        
    # for indexpitch in range (math.floor((3/5) * (mfcc.shape[0]-1)),mfcc.shape[0]): #takes off last 33% of dct- pitch
    #     mfcc[indexpitch] = 0
        
        
    # plt.plot(mfcc[:,40])
    # plt.figure()
    # plt.hist(mfcc)
    # # plt.colorbar()
    # print(mfcc)
    # plt.show()
    for indexvmfcc in range (1, tri_cut-2): #1 to 32 (6) #velocity calced and rows added to mfcc
        row_n = mfcc.shape[0]#bottom row
        mfcc = np.insert(mfcc,row_n,np.zeros(len(mag_frames)),axis=0) #add new row of length mag frames
        for indexvel in range (0, len(mfcc[indexvmfcc])-1): #0 to 279 (for example)
            velocity = mfcc[indexvmfcc+1][indexvel] - mfcc[indexvmfcc-1][indexvel]
            mfcc[row_n][indexvel] = velocity #putting the velocity in the new row
    for indexamfcc in range (1, tri_cut-4): #1 to 30 (6) #acceleration calced and rows added to mfcc
        row_n = mfcc.shape[0]#new bottom row
        mfcc = np.insert(mfcc,row_n,np.zeros(len(mag_frames)),axis=0) #add new row of length mag frames
        for indexacc in range (0, len(mfcc[indexamfcc])-1): #0 to 279 (for example)
            velocity = mfcc[indexamfcc+1][indexacc] - mfcc[indexamfcc-1][indexacc]
            mfcc[row_n][indexacc] = velocity #putting the velocity in the new row
    row_n = mfcc.shape[0] #new bottom row
    mfcc = np.insert(mfcc,row_n,np.zeros(len(mag_frames)),axis=0)
    mfcc[row_n][0] = file_energy
        
    
    #plt.plot(mfcc[:,5])
    #plt.figure()
    
    return mfcc
    #week 3 volecity and accelerations
    #[40,12] from shaping the mfccs, 40 was 8 (number of fbanks) originally but adjusted with acel and vel stuff - called mfcc features
    #12/the number that changes is number of frames. ours is way larger cause they recorded at 8kHz and are quicker at recording less no noise bits

def getEnergy(speech):
    energy = 0
    for index in range (0, len(speech-1)):
        energy = energy + (speech[index])**2
    #print(energy)
    return energy
    

    
bins = melSpacedPoints(0, 8000,34) #cant go above 34 because otherwise the numbers
#in the mel-point diagram get too small and round to 0 which cant be divided
tri_fbanks = triangularFBank(bins)
