# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:34:53 2024

@author: scott
"""


import numpy as np
import plotly.io as pio
import plotly.express as px
import sounddevice as sd
import matplotlib.pyplot as pypl
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
def toMagFrames(speech):
    num_samples = len(speech)
    frame_length = 320 #this is 20ms as the length of speech file is in freq
    num_of_frames = math.floor((num_samples-frame_length)/(frame_length/2)) #calcs the number of frames, so i can create a array of appropraite size
    mag_frames = [None] * (num_of_frames+1) #empty list that can fit the number of frames
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
    triVector = [len(binlist)] #list of length of the amount of bins
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
            triVector[index][k] = triValue
    return triVector

def applyFbankLogDCT(mag_frames):
    mfcc = [None] * len(mag_frames)
    for index in range (len(mag_frames)):
        for k in range (len(tri_fbanks)):
            #matmul every mag frame but each tribank filter
            #then log it, then apply dct equation
            mfcc[index] = np.fft.dct(math.log10(np.matmul(mag_frames[index],tri_fbanks[k])))
            #removing latter half to try and remove the pitch, may need to change what % is being removed
            mfcc[index] = mfcc[index[0:int(len(mfcc[index]/2))]]
    np.save('test', mfcc)
    
bins = melSpacedPoints(0, 8000, 8)
tri_fbanks = triangularFBank(bins)
speechFile, fs = sf.read('look_out.wav', dtype='float32')
toMagFrames(speechFile)



