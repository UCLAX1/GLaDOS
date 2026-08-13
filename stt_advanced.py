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
import torch
import pyaudio
import audioop
import numpy as np
import torch
import time
import queue
import collections
import threading
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

def int16_bytes_to_normalized_float32_ndarray(int16_bytes: bytes) -> np.ndarray:
    return np.frombuffer(int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

def int16_bytes_list_to_normalized_float32_ndarray(int16_bytes_list: list[bytes]) -> np.ndarray:
    return int16_bytes_to_normalized_float32_ndarray(b"".join(int16_bytes_list))


def round_to_nearest(n, m):
    return (n + m - 1) // m * m

def preprocess_text(text: str) -> str:
    # ripped from https://github.com/KoljaB/RealtimeSTT/blob/master/tests/realtimestt_test.py
    # Remove leading whitespaces
    text = text.lstrip()

    #  Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

class Transcriber():

    def __init__(self, device):
        """
        device: either "cuda" or "cpu"
        """

        self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # distil-small is less accurate but around 3x faster
        self.model_size = "distil-large-v3" if self.device == "cuda" else "distil-small.en"
        # self.model_size = "distil-large-v3" if self.device == "cuda" else "distil-large-v3"

        self.faster_whisper_model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        # pretty accurate and about 4-5 times faster
        # self.faster_whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")

        # recognizer = sr.Recognizer()

    def transcribe(self, speech_chunks: list[bytes]) -> str:
        transcribed_text = ""

        speech_chunks_float32: np.ndarray = int16_bytes_list_to_normalized_float32_ndarray(speech_chunks)
        # segments, info = self.faster_whisper_model.transcribe(speech_chunks_float32, language="en", condition_on_previous_text=True)
        segments, info = self.faster_whisper_model.transcribe(speech_chunks_float32, language="en", condition_on_previous_text=False)
        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text

        # return recognizer.recognize_faster_whisper(audio_data, language="en")

class TranscriptionWorker():
    def __init__(self, state_changed_event: threading.Event, transcriber: Transcriber, device, on_transcription_update_callback):
        """
        device: either "cuda" or "cpu"
        """
        self.transcriber = transcriber

        # queue of audio to be transcribed
        self.audio_to_transcribe_queue: queue.Queue[list[bytes]] = queue.Queue()

        self.on_transcription_update_callback = on_transcription_update_callback

        # # may be multiple sentences
        self.latest_transcribed_text = ""

        self.time_taken_to_transcribe = 0.0

        self.state_changed_event = state_changed_event
        self._stop_event = threading.Event()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        self.is_busy = False

    def submit_transcription_request(self, chunks_to_transcribe: list[bytes]):
        self.audio_to_transcribe_queue.put(
            list(chunks_to_transcribe)
        )
    
    def stop(self):
        self._stop_event.set()
        self.thread.join(timeout=1.0)


    def _worker(self):
        while True:
            try:
                speech_chunks = self.audio_to_transcribe_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            transcription_start_time = time.perf_counter()

            self.is_busy = True

            self.latest_transcribed_text = self.transcriber.transcribe(speech_chunks)

            self.is_busy = False

            self.time_taken_to_transcribe = time.perf_counter() - transcription_start_time

            threading.Thread(
                target=self.on_transcription_update_callback,
                args=(self.latest_transcribed_text,)
            ).start()

            self.state_changed_event.set()

class Recorder():

    NUM_CHANNELS=1
    SAMPLE_RATE=16000
    CHUNK_SIZE = 512 # num frames per buffer

    CHUNKS_PER_SECOND = SAMPLE_RATE / CHUNK_SIZE
    SECONDS_PER_CHUNK = CHUNK_SIZE / SAMPLE_RATE

    PRE_AUDIO_CHUNK_BUFFER_DURATION = 0.3 # store 0.3 seconds of chunks
    PRE_AUDIO_CHUNK_BUFFER_SIZE = int(PRE_AUDIO_CHUNK_BUFFER_DURATION * CHUNKS_PER_SECOND) # max num chunks

    SPEECH_PROB_THRESHOLD = 0.5

    def __init__(self, transcriber_device: str, vad_device: str, on_transcription_update_callback):

        self.transcriber_device = transcriber_device
        self.vad_device = vad_device

        self.silero_vad_model: torch.nn.Module = load_silero_vad()
        self.silero_vad_model.to(self.vad_device)

        # when we first detect speech, it's only after a few ms of speech is said, and we need to add that to the buffer
        self.pre_audio_chunks_rolling_buffer: collections.deque[list[bytes]] = collections.deque(
            maxlen=self.PRE_AUDIO_CHUNK_BUFFER_SIZE
        ) 

        # chunks with speech with the pre-audio chunks appended to the front
        self.speech_chunks: list[bytes] = []

        self.audio_queue: queue.Queue = queue.Queue()

        self.state_changed_event = threading.Event()

        self.on_transcription_update_callback = on_transcription_update_callback

        print("loading transcriber...")
        self.transcriber = Transcriber(device=self.transcriber_device)
        print("done loading transcriber")

        self.transcription_worker = TranscriptionWorker(
            state_changed_event=self.state_changed_event,
            transcriber=self.transcriber,
            device=self.transcriber_device,
            on_transcription_update_callback=self.on_transcription_update_callback
        )

        self.time_start = time.time()
        self.time_vad_detects_speech_stop = time.time()
        self.time_vad_first_detects_speech = 0.0
        self.time_submitted_transcription_request = 0.0
        # self.silence_duration = time.time()
        self.time_taken_to_detect_voice = 0.0
        self.post_speech_silence_duration = 1.0 # time to wait after speaking first not detected, in seconds

        self.vad_detects_speech_previous = False
        self.vad_detects_speech = False

        self.vad_detects_speech_start = False
        self.vad_detects_speech_stop = False

        self.is_speaking = False

        self.speech_confidence: float = 0.0
        # self.boundary_detected: bool = False

        self.audio = pyaudio.PyAudio()

        self.vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.vad_thread.start()

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.NUM_CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
            stream_callback=self._on_new_audio_chunk_callback,
        )
    
    def __del__(self):
        self.stream.close()
        self.audio.terminate()
    
    def record(self, timeout=SECONDS_PER_CHUNK):
        self.state_changed_event.wait(timeout=timeout)
        self.state_changed_event.clear()

    def _on_new_audio_chunk_callback(self, audio_chunk: bytes, frame_count: int, time_info: dict, status: int) -> tuple:
        self.audio_queue.put(audio_chunk)
        return (None, pyaudio.paContinue)

    def _submit_transcription_request(self, chunks_to_transcribe: list[bytes]):
        self.transcription_worker.submit_transcription_request(chunks_to_transcribe)

    def _vad_worker(self):

        while True:

            if (self.audio_queue.qsize() > 50):
                raise Exception("audio queue got too big, either hardware is too slow or I am bad at coding")

            try:
                audio_chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                continue

            if not self.is_speaking:
                self.pre_audio_chunks_rolling_buffer.append(audio_chunk)

            if self.is_speaking:
                self.speech_chunks.append(audio_chunk)

            vad_start_time = time.perf_counter()

            audio_chunk_float32: np.ndarray = int16_bytes_to_normalized_float32_ndarray(audio_chunk)

            audio_tensor = torch.from_numpy(audio_chunk_float32).to(self.vad_device)
            self.speech_confidence = self.silero_vad_model(audio_tensor, self.SAMPLE_RATE).item()
            # time.sleep(0.05) # <- if it takes too long to detect speech then the queue will grow infinitely
            # but this is less of a bug and more of a hardware limitation

            self.time_taken_to_detect_voice = time.perf_counter() - vad_start_time

            # rms = audioop.rms(audio_chunk, 2)
            # progress.update(task_id, completed=rms)

            # TODO: endpointing
            # https://arunbaby.com/speech-tech/0035-speech-boundary-detection/
            # see: "What is endpointing and why is a fixed timeout insufficient?"
            # right now, this code uses a fixed timeout of 1 second

            self.vad_detects_speech = self.speech_confidence > self.SPEECH_PROB_THRESHOLD
            self.vad_detects_speech_start = self.vad_detects_speech and not self.vad_detects_speech_previous
            self.vad_detects_speech_stop = not self.vad_detects_speech and self.vad_detects_speech_previous

            if self.vad_detects_speech_start and (not self.is_speaking):
                self.is_speaking = True
                self.time_vad_first_detects_speech = time.time()
                self.speech_chunks.extend(self.pre_audio_chunks_rolling_buffer)

                audio_data = sr.AudioData(b"".join(self.pre_audio_chunks_rolling_buffer), sample_rate=self.SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
                with open("pre-audio.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())

                self.pre_audio_chunks_rolling_buffer.clear()

            if self.vad_detects_speech_stop:
                # falling edge
                self.time_vad_detects_speech_stop = time.time()

            # seconds_of_speech_stored = len(self.speech_chunks) * self.SECONDS_PER_CHUNK
            
            # # submit pre transcription only when the transcription worker is not already transcribing something
            # if self.vad_detects_speech and seconds_of_speech_stored > 1.0 and not self.transcription_worker.is_busy:
            #     # pre submit transcription request
            #     print("submitting pre-transcription request")
            #     self._submit_transcription_request(list(self.speech_chunks))
            #     self.time_submitted_transcription_request = time.time()
            
            # if self.vad_detects_speech_stop:
            #     print("submitting final transcription request")
            #     self._submit_transcription_request(list(self.speech_chunks))
            #     self.time_submitted_transcription_request = time.time()

            if (not self.vad_detects_speech) and self.is_speaking:

                elapsed_silence = time.time() - self.time_vad_detects_speech_stop

                if elapsed_silence > self.post_speech_silence_duration:
                    self.is_speaking = False

                    print("submitting final transcription request")
                    self._submit_transcription_request(self.speech_chunks)
                    self.time_submitted_transcription_request = time.time()

                    # print("writing audio data to file")
                    audio_data = sr.AudioData(b"".join(self.speech_chunks), sample_rate=self.SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
                    with open("microphone-results.wav", "wb") as f:
                        f.write(audio_data.get_wav_data())
                    # print("done writing audio data to file")

                    self.speech_chunks.clear()

            self.vad_detects_speech_previous = self.vad_detects_speech
            self.state_changed_event.set()


transcribed_text = ""

def on_transcription_update(transcribed_text_output: str):
    transcribed_text = transcribed_text_output
    transcribed_text = preprocess_text(transcribed_text)
    full_sentences.append(transcribed_text)


progress = Progress(
    TextColumn("[bold blue]Speech Probability:"),
    BarColumn(bar_width=40),
)
task_id = progress.add_task("vad", total=100)

console = Console()

pre_transcribed_text = ""
full_sentences: list[str] = []

# transcriber_device = "cuda" if torch.cuda.is_available() else "cpu"
# vad_device = "cuda" if torch.cuda.is_available() else "cpu"

transcriber_device = "cpu"
vad_device = "cpu"

print(f"transcriber device: {transcriber_device}")
print(f"vad device: {vad_device}")

recorder = Recorder(
    transcriber_device=transcriber_device,
    vad_device=vad_device,
    on_transcription_update_callback=on_transcription_update,
)

print("Listening...")

with Live(console=console, refresh_per_second=60) as live:
    try:
        while True:

            before_record = time.perf_counter()
            recorder.record()
            time_taken_to_record = time.perf_counter() - before_record

            # DISPLAYING STUFF TO CONSOLE

            transcribed_rich_text = Text()

            for i, sentence in enumerate(full_sentences):
                if i % 2 == 0:
                    #rich_text += Text(sentence, style="bold yellow") + Text(" ")
                    transcribed_rich_text += Text(sentence, style="yellow") + Text(" ")
                else:
                    transcribed_rich_text += Text(sentence, style="cyan") + Text(" ")

            transcribed_rich_text += Text(pre_transcribed_text, style="bold bright_red") + Text(" ")

            """Creates a table combining the bar and the boolean indicator."""
            table = Table.grid(expand=True)
            table.add_column()
            table.add_column(justify="right")

            # Format the boolean status indicator
            if recorder.is_speaking:
                status = "[bold white on green]  SPEAKING  [/bold white on green]"
            else:
                status = "[bold white on dim red]  SILENT    [/bold white on dim red]"

            progress.update(task_id, completed=recorder.speech_confidence * 100)
            table.add_row(progress, status)

            table.add_row(Text(f"size of buffer: {len(recorder.pre_audio_chunks_rolling_buffer)}"))
            table.add_row(Text(f"time taken to transcribe: {recorder.transcription_worker.time_taken_to_transcribe}"))
            table.add_row(Text(f"time taken to detect voice: {recorder.time_taken_to_detect_voice}"))
            table.add_row(Text(f"time taken to record: {time_taken_to_record}"))
            table.add_row(Text(f"size of audio queue: {recorder.audio_queue.qsize()}"))
            table.add_row(Text(f"size of transcription queue: {recorder.transcription_worker.audio_to_transcribe_queue.qsize()}"))
            # table.add_row(Text(f"time since detected speech stop: {silence_duration}"))
            table.add_row(transcribed_rich_text)
            panel = Panel(table, title="[bold]Live VAD Monitor[/bold]", border_style="blue")
            live.update(panel)

            # END DISPLAYING STUFF TO CONSOLE
    except KeyboardInterrupt as e:
        print(e)
    except Exception as e:
        print(e)
    finally:
        exit(1)
    
