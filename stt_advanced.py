# pip install silero-vad
# https://github.com/snakers4/silero-vad 
# https://github.com/snakers4/silero-vad/blob/master/examples/pyaudio-streaming/pyaudio-streaming-examples.ipynb
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import pyaudio
import numpy as np
import torch
import time
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

def generate_display(is_speaking: bool):
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
    return Panel(table, title="[bold]Live VAD Monitor[/bold]", border_style="blue")

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
SPEECH_RELEASE_TIME = 0.5 # time to wait after speaking first not detected, in seconds

stream = audio.open(
    format=pyaudio.paFloat32,
    channels=NUM_CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE,
)

model = load_silero_vad()

data: list[bytes] = []
confidences: list[float] = []

start = time.time()
current_time = start
time_vad_stopped_detecting_speech = start

vad_detecting_speech_previous = False
vad_detecting_speech = False

is_speaking = False

print("Listening...")


with Live(Panel(progress, title="VAD monitor"), refresh_per_second=60) as live:
    while True:
        current_time = time.time()

        frame: bytes = stream.read(CHUNK_SIZE)
        data.append(frame)

        frame_audio: np.ndarray = np.frombuffer(frame, dtype=np.float32)

        speech_confidence: float = model(torch.tensor(frame_audio), SAMPLE_RATE).item()
        confidences.append(speech_confidence)

        progress.update(task_id, completed=speech_confidence * 100)

        # is_vad_activated logic
        # wait 0.5 seconds after done talking to sets is_vad_activated to False
        time_since_vad_stopped_detecting_speech = current_time - time_vad_stopped_detecting_speech

        vad_detecting_speech = speech_confidence > SPEECH_PROB_THRESHOLD

        if vad_detecting_speech:
            is_speaking = True

        elif not vad_detecting_speech and vad_detecting_speech_previous:
            # falling edge
            time_vad_stopped_detecting_speech = current_time

        elif time_since_vad_stopped_detecting_speech > SPEECH_RELEASE_TIME:
            is_speaking = False

        live.update(generate_display(is_speaking))

        vad_detecting_speech_previous = vad_detecting_speech


