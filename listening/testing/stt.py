#%%
# https://www.geeksforgeeks.org/python/python-convert-speech-to-text-and-text-to-speech/
# some examples for using the speech_recognition library:
# https://github.com/Uberi/speech_recognition/blob/master/examples
# docs for speech_recognition:
# https://pypi.org/project/SpeechRecognition/
import speech_recognition as sr
from matplotlib import pyplot as plt
import numpy as np

r = sr.Recognizer()

# print(sr.Microphone.list_microphone_names()[:3])
# print(sr.Microphone.list_microphone_names())

# VVV has to be manually inputted based on the name of the actual microphone in GLADOS
# preferred_mic_name = "Microphone Array (Realtek(R) Au"
# preferred_mic_name = "Microphone (NVIDIA Broadcast)"
# preferred_mic_name = "Microphone Array (Realtek(R) Audio)"
preferred_mic_name = None  # set to mic name substring to select a specific mic, None for system default

preferred_mic_device_index = sr.Microphone.list_microphone_names().index(preferred_mic_name) if preferred_mic_name else None

while True:
    try:
        with sr.Microphone(sample_rate=16000, device_index=preferred_mic_device_index) as source:
            # print("Listening...")
            
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source)
            # text = r.recognize_google(audio, language="en-US")
            text = r.recognize_faster_whisper(audio, language="en")
            text = text.lower()  
            print(text)
            # print("You said:", text)

            array = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            plt.plot(array, color='green', linewidth=1)
            plt.savefig("audio.png")

            if "exit" in text:
                print("Exiting program...")
                break

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Could not understand audio")

    except KeyboardInterrupt:
        print("Program terminated by user")
        break

# %%
