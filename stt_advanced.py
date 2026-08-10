# pip install silero-vad
# https://github.com/snakers4/silero-vad 
# https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import speech_recognition as sr
import pyaudio
import audioop
import numpy as np
import torch
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

progress = Progress(
    TextColumn("[bold blue]Speech Probability:"),
    BarColumn(bar_width=40),
)
task_id = progress.add_task("vad", total=100)

audio = pyaudio.PyAudio()

NUM_CHANNELS=1
SAMPLE_RATE=16000
CHUNK_SIZE = 512 # num frames per buffer

SPEECH_PROB_THRESHOLD = 0.5
FINISHED_SPEAKING_TIMEOUT_DURATION = 1.0 # time to wait after speaking first not detected, in seconds

stream = audio.open(
    format=pyaudio.paInt16,
    channels=NUM_CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

model = load_silero_vad()

recognizer = sr.Recognizer()

prefired_text = ""
text = ""
displayed_text = ""
full_sentences = []

frames: list[bytes] = []
# confidences: list[float] = []

start = time.time()
current_time = start
time_vad_detects_speech_stop = start

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

        frame_audio: np.ndarray = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0

        speech_confidence: float = model(torch.tensor(frame_audio), SAMPLE_RATE).item()
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
        time_since_vad_stopped_detecting_speech = current_time - time_vad_detects_speech_stop

        vad_detects_speech = speech_confidence > SPEECH_PROB_THRESHOLD
        vad_detects_speech_start = vad_detects_speech and not vad_detects_speech_previous
        vad_detects_speech_stop = not vad_detects_speech and vad_detects_speech_previous

        if is_speaking:
            frames.append(frame)

        if vad_detects_speech_start:
            is_speaking = True

        elif vad_detects_speech_stop:
            # falling edge
            time_vad_detects_speech_stop = current_time

        elif is_speaking and not vad_detects_speech and time_since_vad_stopped_detecting_speech > FINISHED_SPEAKING_TIMEOUT_DURATION:
            is_speaking = False

            audio_data = sr.AudioData(b"".join(frames), sample_rate=SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
            frames.clear()

            with open("microphone-results.wav", "wb") as f:
                f.write(audio_data.get_wav_data())

            prefired_text = recognizer.recognize_faster_whisper(audio_data, language="en")
            text = prefired_text
            full_sentences.append(text)




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
        table.add_row(rich_text)
        panel = Panel(table, title="[bold]Live VAD Monitor[/bold]", border_style="blue")
        live.update(panel)

        # END DISPLAYING STUFF TO CONSOLE

        vad_detects_speech_previous = vad_detects_speech


