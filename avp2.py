# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 22:49:12 2024

@author: tommy
"""

import glob
from pathlib import Path
import os
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
### freq
fs = 16000


## magSpec calc
def magAndPhase(SpeechFrame):
    s = np.squeeze(SpeechFrame)
    Hamming_window = s*np.hamming(320)
    Xf = np.fft.fft(Hamming_window)
    shortTimeMag = np.abs(Xf)
    shortTimephase= np.angle(Xf)
    
    return shortTimeMag, shortTimephase
file_count = 0

### Loading in all audio files found in directory
for soundfile in sorted(glob.glob('AVPSAMPLESMONO\\*\\*.wav')):
    r, fs = sf.read(soundfile)
    file_count = file_count+1
    frameLength = 320
    numSamples = len(r)
    numFrames = int(np.floor(numSamples/frameLength))
    print(file_count, soundfile)
    for index in np.arange(0,numSamples-frameLength,int(frameLength/2)):
      startIndex = index # compute first sample of frame
      endIndex = index + frameLength # compute last sample of frame
      shortTimeFrame = r[startIndex:endIndex] # shortTimeFrame = speechFile[firstSample:lastSample]
      magAndPhase(shortTimeFrame) # magSpec, phaseSpec = magAndPhase(shortTimeFrame)
   