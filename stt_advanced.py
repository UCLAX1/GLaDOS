# pip install silero-vad

# https://github.com/snakers4/silero-vad 

# pyaudio microphone example:
# https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb

# RealtimeSTT recording.py:
# https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/core/recording.py

# RealtimeSTT realtime_boundary_detector.py:
# https://github.com/KoljaB/RealtimeSTT/blob/master/RealtimeSTT/core/realtime_boundary_detector.py#L128

# faster whisper:
# https://pypi.org/project/faster-whisper/

# speech_recognition:
# https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import speech_recognition as sr
from faster_whisper import WhisperModel
import pyaudio
import audioop
import numpy as np
import torch
import time
import collections
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

def int16_bytes_to_normalized_float32_ndarray(int16_bytes: bytes) -> np.ndarray:
    return np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def int16_byte_list_to_normalized_float32_ndarray(int16_byte_list: list[bytes]) -> np.ndarray:
    return int16_bytes_to_normalized_float32_ndarray(b"".join(int16_byte_list))


def round_to_nearest(n, m):
    return (n + m - 1) // m * m

progress = Progress(
    TextColumn("[bold blue]Speech Probability:"),
    BarColumn(bar_width=40),
)
task_id = progress.add_task("vad", total=100)

audio = pyaudio.PyAudio()

NUM_CHANNELS=1
SAMPLE_RATE=16000
CHUNK_SIZE = 512 # num frames per buffer

CHUNKS_PER_SECOND = SAMPLE_RATE / CHUNK_SIZE

SPEECH_PROB_THRESHOLD = 0.5
FINISHED_SPEAKING_TIMEOUT_DURATION = 1.0 # time to wait after speaking first not detected, in seconds

stream = audio.open(
    format=pyaudio.paInt16,
    channels=NUM_CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

silero_vad_model = load_silero_vad()

recognizer = sr.Recognizer()
print("loading faster whisper model...")
faster_whisper_model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
print("done loading faster whisper model")

# transcribed_text = ""
pre_transcribed_text = ""
displayed_text = ""
full_sentences = []

speech_frames: list[bytes] = []

# when we first detect speech, it's only after a few ms of speech is said, and we need to add that to the buffer
PRE_AUDIO_FRAME_BUFFER_DURATION = 0.3 # store 0.3 seconds of frames
PRE_AUDIO_FRAME_BUFFER_SIZE = int(PRE_AUDIO_FRAME_BUFFER_DURATION * CHUNKS_PER_SECOND) # max num chunks
pre_audio_frames_rolling_buffer: collections.deque[bytes] = collections.deque(maxlen=PRE_AUDIO_FRAME_BUFFER_SIZE) # rolling buffer of frame chunks

start = time.time()
current_time = start
time_vad_detects_speech_stop = start
time_since_vad_stopped_detecting_speech = start
time_taken_to_transcribe = 0.0

vad_detects_speech_previous = False
vad_detects_speech = False

vad_detects_speech_start = False
vad_detects_speech_stop = False

is_speaking = False

print("Listening...")

console = Console()

with Live(console=console, refresh_per_second=60) as live:
    while True:
        current_time = time.time()

        frame: bytes = stream.read(CHUNK_SIZE)

        frame_float32: np.ndarray = int16_bytes_to_normalized_float32_ndarray(frame)

        speech_confidence: float = silero_vad_model(torch.tensor(frame_float32), SAMPLE_RATE).item()
        # confidences.append(speech_confidence)

        progress.update(task_id, completed=speech_confidence * 100)
        # rms = audioop.rms(frame, 2)
        # progress.update(task_id, completed=rms)

        # TODO: endpointing
        # https://arunbaby.com/speech-tech/0035-speech-boundary-detection/
        # see: "What is endpointing and why is a fixed timeout insufficient?"
        # right now, this code uses a fixed timeout of 1 second

        # VAD DETECTION LOGIC

        # wait 1 seconds after done talking to sets is_vad_activated to False

        vad_detects_speech = speech_confidence > SPEECH_PROB_THRESHOLD
        vad_detects_speech_start = vad_detects_speech and not vad_detects_speech_previous
        vad_detects_speech_stop = not vad_detects_speech and vad_detects_speech_previous
        # MAYBE TODO: implement better boundary detection 
        # so GLADOS can stop talking if someone says "hey glados" or "shut up" or whatever
        # boundary_detected = vad_detects_speech_stop

        if is_speaking:
            speech_frames.append(frame)
        
        if not is_speaking:
            pre_audio_frames_rolling_buffer.append(frame)
        
        if vad_detects_speech_start and (not is_speaking):
            speech_frames.extend(pre_audio_frames_rolling_buffer)

            # audio_data = sr.AudioData(b"".join(pre_audio_frames_rolling_buffer), sample_rate=SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
            # with open("pre-audio.wav", "wb") as f:
            #     f.write(audio_data.get_wav_data())

            pre_audio_frames_rolling_buffer.clear()

        if vad_detects_speech_start:
            is_speaking = True

        if vad_detects_speech_stop:
            # falling edge
            time_vad_detects_speech_stop = current_time

        time_since_vad_stopped_detecting_speech = current_time - time_vad_detects_speech_stop

        if is_speaking and (not vad_detects_speech) and time_since_vad_stopped_detecting_speech > FINISHED_SPEAKING_TIMEOUT_DURATION:
            is_speaking = False

            audio_data = sr.AudioData(b"".join(speech_frames), sample_rate=SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
            with open("microphone-results.wav", "wb") as f:
                f.write(audio_data.get_wav_data())

            transcription_start_time = time.perf_counter()

            speech_frames_float32: np.ndarray = int16_byte_list_to_normalized_float32_ndarray(speech_frames)
            segments, info = faster_whisper_model.transcribe(speech_frames_float32, language="en", condition_on_previous_text=True)
            for segment in segments:
                full_sentences.append(segment.text)

            # text = recognizer.recognize_faster_whisper(audio_data, language="en")
            # full_sentences.append(text)

            time_taken_to_transcribe = time.perf_counter() - transcription_start_time


            speech_frames.clear()


        # END VAD DETECTION LOGIC

        # DISPLAYING STUFF TO CONSOLE

        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            if i % 2 == 0:
                #rich_text += Text(sentence, style="bold yellow") + Text(" ")
                rich_text += Text(sentence, style="yellow") + Text(" ")
            else:
                rich_text += Text(sentence, style="cyan") + Text(" ")

        # new_displayed_text = rich_text.plain

        # if new_displayed_text != displayed_text:
        #     displayed_text = new_displayed_text
            # panel = Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green")
            # live.update(panel)

        """Creates a table combining the bar and the boolean indicator."""
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify="right")

        # Format the boolean status indicator
        if is_speaking:
            status = "[bold white on green]  SPEAKING  [/bold white on green]"
        else:
            status = "[bold white on dim red]  SILENT    [/bold white on dim red]"

        table.add_row(progress, status)
        table.add_row(Text(f"size of buffer: {len(pre_audio_frames_rolling_buffer)}"))
        table.add_row(Text(f"time taken to transcribe: {time_taken_to_transcribe}"))
        # table.add_row(Text(f"time since detected speech stop: {time_since_vad_stopped_detecting_speech}"))
        table.add_row(rich_text)
        panel = Panel(table, title="[bold]Live VAD Monitor[/bold]", border_style="blue")
        live.update(panel)

        # END DISPLAYING STUFF TO CONSOLE

        vad_detects_speech_previous = vad_detects_speech


