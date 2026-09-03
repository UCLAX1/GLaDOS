# listening

folder for audio detection for GLaDOS

the main file is stt_advanced.py

Recorder is the main class to use. It records audio and transcribes it. You can define callbacks to decide what happens when the final transcribed text is ready or when the realtime-transcribed text is ready.

## Basic Usage:

```python
from recorder import Recorder
import time

def on_transcription_update:
    # your code here
    pass

def on_realtime_transcription_update:
    # your code here
    pass

if __name__ == "__main__":

    recorder = Recorder(
        transcriber_device="gpu",
        vad_device="gpu",
        language="en",
        enable_realtime_transcription=True,
        enable_early_transcription=True,
        transcriber_model_type="distil-large-v3",
        on_transcription_update_callback=on_transcription_update,
        realtime_transcriber_model_type="distil-small.en",
        on_realtime_transcription_update_callback=on_realtime_transcription_update,
    )

    recorder.start()

    while True:
        # do whatever
        time.sleep(0.01) # optional
```

There is just about no AI slop in this file. It is mainly referenced from two projects:

https://github.com/KoljaB/RealtimeSTT/
https://github.com/n1teshy/py-listener/