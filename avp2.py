

import glob
from pathlib import Path
import os
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
### freq
frequency = 16000


## magSpec calc
def magAndPhase(SpeechFrame):
    speech = np.squeeze(SpeechFrame)
    hamming_window = speech*np.hamming(320)
    fft = np.fft.fft(hamming_window)
    shortTimeMag = np.abs(fft)
    shortTimephase = np.angle(fft)
    
    return shortTimeMag, shortTimephase

file_count = 0

### Loading in all audio files found in directory
for soundfile in sorted(glob.glob('AVPSAMPLESMONO\\*\\*.wav')):
    r, frequency = sf.read(soundfile)
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
   