import os
import pytest
import numpy as np
import scipy.io.wavfile as wav

# You may need to install librosa if not already present
try:
    import librosa
except ImportError:
    librosa = None

def onset_detection(audio_path):
    if librosa is None:
        raise ImportError("librosa is required for onset detection. Please install it with 'pip install librosa'.")
    y, sr = librosa.load(audio_path, sr=None)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    return onsets

def test_onset_detection_on_demo():
    demo_wav = 'demo_performance.wav'
    assert os.path.exists(demo_wav), f"Demo file {demo_wav} not found."
    onsets = onset_detection(demo_wav)
    print(f"Detected onsets (seconds): {onsets.tolist()}")
    assert isinstance(onsets, np.ndarray)
    assert len(onsets) > 0, "No onsets detected in demo file."

def test_onset_detection_on_any_wav():
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    assert wav_files, "No .wav files found in the current directory."
    for wav_file in wav_files:
        onsets = onset_detection(wav_file)
    print(f"{wav_file}: {len(onsets)} onsets detected. Onset times (s): {onsets.tolist()}")
    assert isinstance(onsets, np.ndarray)
    assert len(onsets) > 0, f"No onsets detected in {wav_file}."
