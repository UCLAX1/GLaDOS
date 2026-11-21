"""
pip install sounddevice numpy scipy
"""
"""
import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# --- 1. Record and Save output.wav ---
fs = 44100        # Sample rate (Hz)
seconds = 5       # Duration of recording
wav_filename = "output.wav"

print("Recording for 5 seconds...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

wavfile.write(wav_filename, fs, recording)
print(f"Saved recording to: {os.path.abspath(wav_filename)}\n")


# --- 2. Analyze output.wav for Sounds Over 70 dB ---
def find_loud_db(wav_path, db_threshold, calibration_offset=95.0, chunk_ms=50):
    fs, data = wavfile.read(wav_path)
    
    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Normalize to [-1.0, 1.0]
    data = data.astype(np.float32) / 32768.0
    
    # Split audio into small chunks (50ms is standard for dB meters)
    chunk_size = int(fs * (chunk_ms / 1000.0))
    num_chunks = len(data) // chunk_size
    chunks = data[:num_chunks * chunk_size].reshape(-1, chunk_size)
    
    # Calculate RMS loudness for every chunk
    rms = np.sqrt(np.mean(chunks**2, axis=1))
    
    # Convert RMS to dBFS, then add calibration offset to estimate dB SPL
    dbfs = 20 * np.log10(rms + 1e-9)
    estimated_db = dbfs + calibration_offset
    
    # Find chunks exceeding 70 dB
    loud_indices = np.where(estimated_db > db_threshold)[0]
    timestamps = (loud_indices * chunk_ms) / 1000.0
    
    # Find peak dB and the exact second it occurred
    peak_index = np.argmax(estimated_db)
    peak_db = estimated_db[peak_index]
    peak_time = (peak_index * chunk_ms) / 1000.0
    
    return timestamps, peak_db, peak_time


# Run the check
db_threshold=80.0 #value it must pass to be of interest to Glados


timestamps, peak_db, peak_time = find_loud_db(wav_filename, db_threshold)
print(f"Peak volume recorded: {peak_db:.1f} dB")
print(f"Peak volume recorded: {peak_db:.1f} dB at {peak_time:.2f} seconds")

if len(timestamps) > 0:
    print(f"🔊 Sound over {db_threshold} dB detected at seconds:", np.round(timestamps, 2))
else:
    print(f"No sounds over {db_threshold} dB detected.")

"""

import os
from datetime import datetime
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# --- Settings ---
fs = 44100              # Sample rate (Hz)
seconds = 1             # Duration of each recording window
wav_filename = "output.wav"
db_threshold = 90.0     # Value it must pass to be of interest to GLaDOS


# --- Analysis Function ---
def find_loud_db(wav_path, db_threshold, calibration_offset=95.0, chunk_ms=50):
    fs, data = wavfile.read(wav_path)
    
    # Convert stereo to mono if needed
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Normalize to [-1.0, 1.0]
    data = data.astype(np.float32) / 32768.0
    
    # Split audio into small chunks (50ms is standard for dB meters)
    chunk_size = int(fs * (chunk_ms / 1000.0))
    num_chunks = len(data) // chunk_size
    chunks = data[:num_chunks * chunk_size].reshape(-1, chunk_size)
    
    # Calculate RMS loudness for every chunk
    rms = np.sqrt(np.mean(chunks**2, axis=1))
    
    # Convert RMS to dBFS, then add calibration offset to estimate dB SPL
    dbfs = 20 * np.log10(rms + 1e-9)
    estimated_db = dbfs + calibration_offset
    
    # Find chunks exceeding threshold
    loud_indices = np.where(estimated_db > db_threshold)[0]
    timestamps = (loud_indices * chunk_ms) / 1000.0
    
    # Find peak dB and the exact second it occurred
    peak_index = np.argmax(estimated_db)
    peak_db = estimated_db[peak_index]
    peak_time = (peak_index * chunk_ms) / 1000.0
    
    return timestamps, peak_db, peak_time


# --- Continuous Monitoring Loop ---
print(f"Starting continuous monitoring (Threshold: {db_threshold} dB).")
print(f"Saving rolling buffer to: {os.path.abspath(wav_filename)}")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] Listening for {seconds} seconds...")
        
        # 1. Record and overwrite output.wav
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wavfile.write(wav_filename, fs, recording)

        # 2. Analyze the recording
        timestamps, peak_db, peak_time = find_loud_db(wav_filename, db_threshold)
        print(f"Peak volume recorded: {peak_db:.1f} dB at {peak_time:.2f} seconds")

        if len(timestamps) > 0:
            print(f"🔊 Sound over {db_threshold} dB detected at seconds:", np.round(timestamps, 2))
            # --> Trigger GLaDOS action or transcription here! <--
        else:
            print(f"No sounds over {db_threshold} dB detected.")
        
        print("-" * 45)

except KeyboardInterrupt:
    print("\nMonitoring stopped.")
