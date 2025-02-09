# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:03:02 2024

    This file is responsible for processing audio, visual and combined data.

@author: tommy
"""

from scipy.interpolate import CubicSpline
import soundfile as sf
import numpy as np
import glob
from pathlib import Path
import os
from main import applyFbankLogDCT, toMagFrames, getEnergy
from visual_preprocessing import detect_face, dct8by8, detect_mouth, get_frame, bitmap_smooth
from moviepy.video.io.VideoFileClip import VideoFileClip

names = {
    0: "Muneeb",
    1: "Zachary",
    2: "Sebastian",
    3: "Danny",
    4: "Louis",
    5: "Ben",
    6: "Seb",
    7: "Ryan",
    8: "Krish",
    9: "Christopher",
    10: "Kaleb",
    11: "Konark",
    12: "Amelia",
    13: "Emilija",
    14: "Naima",
    15: "Leo",
    16: "Noah",
    17: "Josh",
    18: "Joey",
    19: "Kacper"
    }


def process_audio_files():
    """
        The function generates all the npy files containing mfcc values and stores them in their own
        directories, which is determined by name.

        Args:

        Returns:
            Returns Nothing.
    """
    for name in names:
        index = 0
        name = names[name]
        # For every .wav file in sample folder
        for i, soundfile in enumerate(sorted(glob.glob(f'AVPSAMPLESSD\\{name}\\*.wav'))):
            speech_file, frequency = sf.read(soundfile, dtype='float32')
            noise, freq = sf.read('noise.wav', dtype='float32')
            # Matching the length of noise to that of the speech file
            noise = noise[:speech_file.shape[0]]

            # Adding noise distortion
            if index < 15:
                noise_power = np.mean(noise**2)
                speech_power = np.mean(speech_file**2)
                amplification = np.sqrt((speech_power/noise_power)*(10**(-(10/10))))
                speech_file = speech_file+(amplification*noise)

            if 15 < index < 25:
                noise_power = np.mean(noise**2)
                speech_power = np.mean(speech_file**2)
                amplification = np.sqrt((speech_power/noise_power)*(10**(-(20/10))))
                speech_file = speech_file+(amplification*noise)

            # Applying preprocessing feature extraction
            mag_frames = toMagFrames(speech_file)
            file_energy = getEnergy(speech_file)
            mfcc_file = applyFbankLogDCT(mag_frames, file_energy)

            # Saving mfccs by name and number
            np.save(f'mfccs\\{name}\\{name}_{index}', mfcc_file)
            index += 1


def process_visual_data(option_critic):
    """
        The function generates all the npy files containing dct values and stores them in their own
        directories, which is determined by name.

        Args:

        Returns:
            Returns Nothing.
    """
    for name in names:
        index = 0
        name = names[name]
        for video in enumerate(sorted(glob.glob(rf'C:\Users\tommy\Desktop\AudiovisualLabs\AVPSum\VisualClips\{name}\*.mp4'))):
            # Get individual frames of the clip
            frames = get_frame(video)
            vid_coefficients = []

            for frame in frames:
                # Detect a face
                faces = detect_face(frame)
                # From the face, detect the mouth
                detected_mouth = detect_mouth(faces)
                # Conditional statement to avoid None values being appended
                if detected_mouth is None:
                    break

                if option_critic == 'bitmap':
                    data = bitmap_smooth(detected_mouth)

                    vid_coefficients.append(data)

                elif option_critic == 'dct':
                    data = dct8by8(detected_mouth)

                    vid_coefficients.append(data)
            if option_critic == 'bitmap':
                if os.path.exists(f'bitmapmfcc\\{name}') is False:
                    os.mkdir(f'bitmapmfcc\\{name}')
                np.save(f'bitmapmfcc\\{name}\\{name}_{index}', vid_coefficients)
            else:
                if os.path.exists(f'dct\\{name}') is False:
                    os.mkdir(f'dct\\{name}')
                np.save(f'dct\\{name}\\{name}_{index}', vid_coefficients)
            index += 1


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
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[0]))

    for feature_dim in range(visual_feat.shape[0]):
        cubic_spline = CubicSpline(np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim])
        visual_feat_interp[:, feature_dim] = cubic_spline(np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps))

    return visual_feat_interp


def process_mp4():

    for name in names:
        vid_no = 1
        audio_no = 1
        name = names[name]

        for video in enumerate(sorted(glob.glob(f'AVPSAV\\{name}\\*.mp4'))):
            # goes through each vid and extracts the audio as .wav
            if vid_no != 10:
                input_file = VideoFileClip(f"AVPSAV\\{name}\\{name}00{vid_no}.mp4")
                audio1 = input_file.audio

                audio1.write_audiofile(f"VisualWav\\{name}\\{name}00{vid_no}.wav", fps=16000)
                # Extract the audio and save it as an MP3 file
                if os.path.exists(f'VisualWav\\{name}') is False:
                    os.mkdir(f'VisualWav\\{name}')

                vid_no += 1
            else:
                input_file = VideoFileClip(f"AVPSAV\\{name}\\{name}0{vid_no}.mp4")
                audio1 = input_file.audio

                audio1.write_audiofile(f"VisualWav\\{name}\\{name}0{vid_no}.wav", fps=16000)
                vid_no += 1

        for i, soundfile in enumerate(sorted(glob.glob(f"VisualWav\\{name}\\*.wav"))):
            speech_file_16k, frequency = sf.read(soundfile, dtype='float32')
            speech_file_16k = speech_file_16k.mean(axis=1)

            # frequency should be 48k here, resample to 16k
            # speechFile_16k = librosa.resample(speechFile_48k, orig_sr=frequency, target_sr=16000)

            # | Applying preprocessing feature extraction
            mag_frames = toMagFrames(speech_file_16k)
            file_energy = getEnergy(speech_file_16k)
            mfcc_file = applyFbankLogDCT(mag_frames, file_energy)

            # this is where we need to do the visual stuff so that we can append it to the npy file before saving
            # hopefully this then grabs the frames of the relevant mp4
            if audio_no != 10:
                vis_frames = get_frame(f'AVPSAV\\{name}\\{name}00{audio_no}.mp4')
                if len(vis_frames) == 0:
                    print("not found")
                vis_frames = np.array(vis_frames)

            else:
                vis_frames = get_frame(f'AVPSAV\\{name}\\{name}0{audio_no}.mp4')
                vis_frames = np.array(vis_frames)

            # this returns an interpolated array where the visual frames line up to the mccFiles (audios)
            print(mfcc_file.shape)
            # interp_vis = visual_feature_interp(vis_frames, mfccFile)
            # the frames in interp_vis are the ones we want to make into bitmaps and dct

            vis_coefficients = np.zeros(4096)
            vis_bitmaps = np.zeros(4096)
            face_num = 1

            for frame in vis_frames:
                print(f"Mouth detection: {name} {face_num}")
                faces = detect_face(frame)
                detected_mouth = detect_mouth(faces)
                c = dct8by8(detected_mouth)
                c2 = bitmap_smooth(detected_mouth)

                c_1d = c.ravel()
                c2_1d = c2.ravel()

                row_n = vis_coefficients.shape[0]  # new bottom row
                vis_coefficients = np.insert(vis_coefficients, row_n, c_1d, axis=0)

                vis_bitmaps = np.insert(vis_bitmaps, row_n, c2_1d, axis=0)

                face_num += 1

            row_n = mfcc_file.shape[0]  # bottom row
            mfcc_file = np.insert(mfcc_file, row_n, vis_coefficients, axis=0)
            row_n = mfcc_file.shape[0]  # new bottom row
            mfcc_file = np.insert(mfcc_file, row_n, vis_bitmaps, axis=0)
            # so now we have our audio mfcc array with 2 extra rows: 1 with the dcts and 1 with bitmaps
            print(mfcc_file.shape)

            # | Saving mfccs by name and number
            if os.path.exists(f'VisualAudMfcc\\{name}') is False:
                os.mkdir(f'VisualAudMfcc\\{name}')
            np.save(f'VisualAudMfcc\\{name}\\{name}_{audio_no}', mfcc_file)
            audio_no += 1


def get_max_dimensionality(option_critic):
    """
    Returns the maximum dimensions for visual, audio, or combined npy files. This function is crucial for padding
    later.

    Args:
        option_critic: String, tells the function which set of dimensions to calculate the maximum arguments for.

    Returns:
        A set of integers containing the maximum value for each dimension.
    """

    if option_critic == 'visual':
        max_visual2 = max_visual1 = max_visual0 = 0
        for mfcc_file in sorted(glob.glob('bitmapmfcc/*/*.npy')):
            mfcc_data = np.load(mfcc_file)
            if len(mfcc_data) != 0:
                if mfcc_data.shape[0] > max_visual0:
                    max_visual0 = mfcc_data.shape[0]
                if mfcc_data.shape[1] > max_visual1:
                    max_visual1 = mfcc_data.shape[1]
                if mfcc_data.shape[2] > max_visual2:
                    max_visual2 = mfcc_data.shape[2]
        return [max_visual0, max_visual1, max_visual2]

    elif option_critic == 'audio':
        max_audio1 = max_audio0 = 0

        for mfcc_file in sorted(glob.glob('mfccs/*/*.npy')):
            mfcc_data = np.load(mfcc_file)
            if mfcc_data.shape[0] > max_audio0:
                max_audio0 = mfcc_data.shape[0]
            if mfcc_data.shape[1] > max_audio1:
                max_audio1 = mfcc_data.shape[1]
        return [max_audio0, max_audio1]

    elif option_critic == 'combined':
        max_combine0 = max_combine1 = 0
        for mfcc_file in sorted(glob.glob('audiovisualfeatures/mfccs/*/*.npy')):
            mfcc_data = np.load(mfcc_file)
            # print(mfcc_data[1].shape)
            if mfcc_data.shape[0] > max_combine0:
                max_combine0 = mfcc_data.shape[0]
            if mfcc_data.shape[1] > max_combine1:
                max_combine1 = mfcc_data.shape[1]

        return [max_combine0, max_combine1]
    else:
        raise Exception('Invalid option, please choose out of the following: ["visual","audio","combined"].')


def visual_npy(vis_dimensions):
    """
    This function loads in all the visual npy files, adds padding and then appends to a data array. The file name
    for the npy file is split so that the name of the person is added to a label array.

    Args:
        max_visual0, max_visual1, max_visual2: Provides the maximum dimensionality of the visual files for smooth
        padding.

    Returns:
        visual_arr: Contains all the visual npy file data after padding for all visual files.
        visual_labels: Contains the split labels containing the names of the files.
    """

    visual_arr = []
    visual_labels = []

    for visual_file in sorted(glob.glob('bitmapmfcc/*/*.npy')):
        # load the mfcc data
        visual_data = np.load(visual_file)
        if len(visual_data) != 0:

            # padding the mfcc data to ensure that the mfccs are the same dimensions, using the previously calculated
            # max lengths for both dimensions
            mfcc_data = np.pad(visual_data, ((0, vis_dimensions[0]-visual_data.shape[0]),
                                             (0, vis_dimensions[1]-visual_data.shape[1]),
                                             (0, vis_dimensions[2]-visual_data.shape[2])))

            # Appending this to training data
            visual_arr.append(mfcc_data)

            # Getting the file name
            stem_file_name = Path(os.path.basename(visual_file)).stem

            # Splitting the file by _ to get the name rather than number
            label = stem_file_name.split('_')

            # Appending the name to array of labels to match the classes
            visual_labels.append(label[0])

    visual_arr = np.array(visual_arr)

    for val in range(len(visual_arr)):
        visual_arr[val] = visual_arr[val] / np.max(visual_arr[val])

    visual_labels = np.array(visual_labels)

    return visual_arr, visual_labels


def audio_npy(aud_dimensions):
    """
    This function loads in all the audio npy files, adds padding and then appends to a data array. The file name
    for the npy file is split so that the name of the person is added to a label array.

    Args:
        max_audio0, max_audio1: Provides the maximum dimensionality of the audio files for smooth padding.

    Returns:
        audio_arr: Contains all the audio npy file data after padding for all audio files.
        audio_labels: Contains the split labels containing the names of the files.
    """

    audio_arr = []
    audio_labels = []

    for audio_file in sorted(glob.glob('mfccs/*/*.npy')):
        audio_data = np.load(audio_file)
        mfcc_data = np.pad(audio_data, ((0, aud_dimensions[0]-audio_data.shape[0]),
                                        (0, aud_dimensions[1]-audio_data.shape[1])))
        audio_arr.append(mfcc_data)
        stem_file_name = Path(os.path.basename(audio_file)).stem
        label = stem_file_name.split('_')
        audio_labels.append(label[0])

    audio_arr = np.array(audio_arr)
    audio_labels = np.array(audio_labels)
    # Normalising
    for val in range(len(audio_arr)):
        audio_arr[val] = audio_arr[val] / np.max(audio_arr[val])

    return audio_arr, audio_labels


def combined_npy(combined_dimensions):
    """
    This function loads in all the combined npy files, adds padding and then appends to a data array. The file name
    for the npy file is split so that the name of the person is added to a label array.

    Args:
        max_combine0, max_combine1: Provides the maximum dimensionality of the combined files for smooth padding.

    Returns:
        combined_arr : Contains all the combined npy file data after padding for all audio files.
        combined_labels : Contains the split labels containing the names of the files.
    """

    combined_labels = []
    combined_arr = []

    for combined_file in sorted(glob.glob('audiovisualfeatures/mfccs/*/*.npy')):
        combined_data = np.load(combined_file)
        print(combined_file)
        mfcc_data = np.pad(combined_data, ((0, combined_dimensions[0]-combined_data.shape[0]),
                                           (0, combined_dimensions[1]-combined_data.shape[1])))
        combined_arr.append(mfcc_data)

        # complete_combine.extend([combined_arr_1,combined_arr_2,combined_arr_3])
        # complete_combine = np.array(complete_combine)
        # print(complete_combine.shape)

        stem_file_name = Path(os.path.basename(combined_file)).stem
        label = stem_file_name.split('_')
        combined_labels.append(label[0])

    combined_arr = np.array(combined_arr)
    combined_labels = np.array(combined_labels)

    # Normalising data
    for val in range(len(combined_arr)):
        combined_arr[val] = combined_arr[val] / np.max(combined_arr[val])

    return combined_arr, combined_labels
