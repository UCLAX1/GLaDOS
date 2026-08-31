# listening

folder for audio detection for GLaDOS

the main file is stt_advanced.py

Recorder is the main class to use. It records audio and transcribes it. You can define callbacks to decide what happens when the final transcribed text is ready or when the realtime-transcribed text is ready.

There is just about no AI slop in this file. It is mainly referenced from two projects:

https://github.com/KoljaB/RealtimeSTT/
https://github.com/n1teshy/py-listener/