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

    def __init__(
        self,
        device: str,
        model_type: str,
        language=None,
    ):
        """
        device: either "cuda" or "cpu"
        """

        self.device = device

        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.model_type = model_type

        self.language = language

        self.faster_whisper_model = WhisperModel(self.model_type, device=self.device, compute_type=self.compute_type)

        # pretty accurate and about 4-5 times faster
        # self.faster_whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")

        # recognizer = sr.Recognizer()

    def transcribe(self, speech_chunks: list[bytes]) -> str:
        transcribed_text = ""

        speech_chunks_float32: np.ndarray = int16_bytes_list_to_normalized_float32_ndarray(speech_chunks)
        # segments, info = self.faster_whisper_model.transcribe(speech_chunks_float32, language=self.language, condition_on_previous_text=True)
        segments, info = self.faster_whisper_model.transcribe(speech_chunks_float32, language=self.language, condition_on_previous_text=False)

        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text

        # return recognizer.recognize_faster_whisper(audio_data, language="en")

class TranscriptionWorker():
    def __init__(
        self,
        state_changed_event: threading.Event,
        transcriber: Transcriber,
        device,
        on_transcription_update_callback,
        enable_early_transcription = False,
    ):
        """
        device: either "cuda" or "cpu"
        """
        self.transcriber = transcriber
        self.enable_early_transcription = enable_early_transcription

        # queue of audio to be transcribed
        self.audio_to_transcribe_queue: queue.Queue[list[bytes]] = queue.Queue()

        if self.enable_early_transcription:
            self.should_keep_queue: queue.Queue[bool] = queue.Queue()

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

            if self.enable_early_transcription:
                # --- EARLY TRANSCRIPTION LOGIC --- 
                initially_awaiting_keep = self.should_keep_queue.empty()

                if not initially_awaiting_keep:
                    should_keep = self.should_keep_queue.get_nowait()
                    if not should_keep:
                        continue
                # --- END EARLY TRANSCRIPTION LOGIC --- 

            transcription_start_time = time.time()

            self.is_busy = True

            self.latest_transcribed_text = self.transcriber.transcribe(speech_chunks)

            self.is_busy = False

            self.time_taken_to_transcribe = time.time() - transcription_start_time

            if self.enable_early_transcription:
                # --- MORE EARLY TRANSCRIPTION LOGIC --- 
                if initially_awaiting_keep:
                    try:
                        should_keep = self.should_keep_queue.get(timeout=4)
                    except queue.Empty:
                        raise Exception("FATAL ERROR: Timeout reached, should_keep queue remained empty after transcription completed")
                    if not should_keep:
                        continue
                # --- END MORE EARLY TRANSCRIPTION LOGIC --- 
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

    def __init__(
        self,
        transcriber_device: str,
        vad_device: str,
        on_transcription_update_callback,
        on_realtime_transcription_update_callback,
        transcriber_model_type="distil-large-v3",
        realtime_transcriber_model_type="tiny",
        enable_realtime_transcription=False,
        enable_early_transcription=True,
        language=None
    ):

        self.enable_realtime_transcription = enable_realtime_transcription
        self.enable_early_transcription = enable_early_transcription
        self.language = language

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

        self.audio_queue: queue.Queue[bytes] = queue.Queue()

        self.state_changed_event = threading.Event()

        self.on_transcription_update_callback = on_transcription_update_callback
        self.on_realtime_transcription_update_callback = on_realtime_transcription_update_callback

        print("loading transcriber...")
        self.transcriber = Transcriber(device=self.transcriber_device, model_type=transcriber_model_type, language=self.language)
        print("done loading transcriber")

        self.transcription_worker = TranscriptionWorker(
            state_changed_event=self.state_changed_event,
            transcriber=self.transcriber,
            device=self.transcriber_device,
            on_transcription_update_callback=self.on_transcription_update_callback,
            enable_early_transcription=self.enable_early_transcription,
        )

        if self.enable_realtime_transcription:
            print("loading realtime transcriber...")
            self.realtime_transcriber = Transcriber(device=self.transcriber_device, model_type=realtime_transcriber_model_type, language=self.language)
            print("done loading realtime transcriber")

            self.realtime_transcription_worker = TranscriptionWorker(
                state_changed_event=self.state_changed_event,
                transcriber=self.realtime_transcriber,
                device=self.transcriber_device,
                on_transcription_update_callback=self.on_realtime_transcription_update_callback,
                enable_early_transcription=False,
            )

        self.time_start = time.time()
        self.time_vad_detects_speech_stop = 0.0
        self.time_vad_first_detects_speech = 0.0
        self.time_submitted_transcription_request = 0.0
        self.time_submitted_realtime_transcription_request = 0.0
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
    
    def _silero_vad_model(self, audio_chunk: bytes) -> float:
        audio_chunk_float32: np.ndarray = int16_bytes_to_normalized_float32_ndarray(audio_chunk)
        audio_tensor = torch.from_numpy(audio_chunk_float32).to(self.vad_device)
        return self.silero_vad_model(audio_tensor, self.SAMPLE_RATE).item()
    
    def _keep_this_early_transcription(self):
        self.transcription_worker.should_keep_queue.put(True)

    def _discard_this_early_transcription(self):
        self.transcription_worker.should_keep_queue.put(False)

    def _submit_early_transcription_request(self, chunks_to_transcribe: list[bytes]):
        self.transcription_worker.submit_transcription_request(chunks_to_transcribe)
        self.time_submitted_transcription_request = time.time()

    def _submit_final_transcription_request(self, chunks_to_transcribe: list[bytes]):
        self.transcription_worker.submit_transcription_request(chunks_to_transcribe)
        self.time_submitted_transcription_request = time.time()

    def _submit_realtime_transcription_request(self, chunks_to_transcribe: list[bytes]):
        self.realtime_transcription_worker.submit_transcription_request(chunks_to_transcribe)
        self.time_submitted_realtime_transcription_request = time.time()
    
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

            vad_start_time = time.time()

            self.speech_confidence = self._silero_vad_model(audio_chunk)
            # time.sleep(0.05) # <- if it takes too long to detect speech then the queue will grow infinitely
            # but this is less of a bug and more of a hardware limitation

            self.time_taken_to_detect_voice = time.time() - vad_start_time

            # rms = audioop.rms(audio_chunk, 2)

            # TODO: endpointing
            # https://arunbaby.com/speech-tech/0035-speech-boundary-detection/
            # see: "What is endpointing and why is a fixed timeout insufficient?"
            # right now, this code uses a fixed timeout of 1 second

            self.vad_detects_speech = self.speech_confidence > self.SPEECH_PROB_THRESHOLD
            self.vad_detects_speech_start = self.vad_detects_speech and not self.vad_detects_speech_previous
            self.vad_detects_speech_stop = not self.vad_detects_speech and self.vad_detects_speech_previous
        
            if self.enable_early_transcription:
                if self.vad_detects_speech_start and self.is_speaking:
                    self._discard_this_early_transcription()

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

            seconds_of_speech_stored = len(self.speech_chunks) * self.SECONDS_PER_CHUNK
            
            if self.enable_realtime_transcription:
                # submit realtime transcription only when the realtime transcription worker is not already transcribing something
                if self.vad_detects_speech and seconds_of_speech_stored > 1.0 and not self.realtime_transcription_worker.is_busy:
                    self._submit_realtime_transcription_request(self.speech_chunks)
            
            if self.enable_early_transcription:
                time_since_last_submitted_transcription_request = time.time() - self.time_submitted_transcription_request

                if self.vad_detects_speech_stop \
                    and time_since_last_submitted_transcription_request > 0.1:
                    # and seconds_of_speech_stored > 0.5:

                    # print("submitting early transcription request")
                    self._submit_early_transcription_request(self.speech_chunks)

            if (not self.vad_detects_speech) and self.is_speaking:

                elapsed_silence = time.time() - self.time_vad_detects_speech_stop

                if elapsed_silence > self.post_speech_silence_duration:
                    self.is_speaking = False

                    if self.enable_early_transcription:
                        self._keep_this_early_transcription()

                    if not self.enable_early_transcription:
                        # print("submitting final transcription request")
                        self._submit_final_transcription_request(self.speech_chunks)

                    # print("writing audio data to file")
                    audio_data = sr.AudioData(b"".join(self.speech_chunks), sample_rate=self.SAMPLE_RATE, sample_width=2) # 2 for 2 byte, 16 bit ints
                    with open("microphone-results.wav", "wb") as f:
                        f.write(audio_data.get_wav_data())
                    # print("done writing audio data to file")

                    self.speech_chunks.clear()

            self.vad_detects_speech_previous = self.vad_detects_speech
            self.state_changed_event.set()


# transcribed_text = ""
realtime_transcribed_text = ""

def on_transcription_update(transcribed_text_output: str):
    global realtime_transcribed_text

    transcribed_text = preprocess_text(transcribed_text_output)
    full_sentences.append(transcribed_text)
    realtime_transcribed_text = ""

def on_realtime_transcription_update(realtime_transcribed_text_output: str):
    global realtime_transcribed_text
    realtime_transcribed_text = preprocess_text(realtime_transcribed_text_output)

progress = Progress(
    TextColumn("[bold blue]Speech Probability:"),
    BarColumn(bar_width=40),
)
task_id = progress.add_task("vad", total=100)

console = Console()

full_sentences: list[str] = []

transcriber_device = "cuda" if torch.cuda.is_available() else "cpu"
vad_device = "cuda" if torch.cuda.is_available() else "cpu"

# transcriber_device = "cpu"
# vad_device = "cpu"

print(f"transcriber device: {transcriber_device}")
print(f"vad device: {vad_device}")

# distil-small is less accurate but around 3x faster
# model_type = "distil-large-v3" if self.device == "cuda" else "distil-small.en"
# model_type = "distil-medium.en"
model_type = "distil-large-v3"

realtime_model_type = "distil-small.en"

recorder = Recorder(
    transcriber_device=transcriber_device,
    vad_device=vad_device,
    language="en",
    enable_realtime_transcription=True,
    # enable_realtime_transcription=False,
    enable_early_transcription=True,
    # enable_early_transcription=False,
    transcriber_model_type=model_type,
    on_transcription_update_callback=on_transcription_update,
    realtime_transcriber_model_type=realtime_model_type,
    on_realtime_transcription_update_callback=on_realtime_transcription_update,
)

print("Listening...")

with Live(console=console, refresh_per_second=60) as live:
    try:
        while True:

            before_record = time.time()
            recorder.record()
            time_taken_to_record = time.time() - before_record

            # DISPLAYING STUFF TO CONSOLE

            transcribed_rich_text = Text()

            for i, sentence in enumerate(full_sentences):
                if i % 2 == 0:
                    #rich_text += Text(sentence, style="bold yellow") + Text(" ")
                    transcribed_rich_text += Text(sentence, style="yellow") + Text(" ")
                else:
                    transcribed_rich_text += Text(sentence, style="cyan") + Text(" ")

            transcribed_rich_text += Text(realtime_transcribed_text, style="bold bright_red") + Text(" ")

            """Creates a table combining the bar and the boolean indicator."""
            table = Table.grid(expand=True)
            table.add_column()
            table.add_column(justify="right")

            # Format the boolean status indicator
            if recorder.is_speaking:
                status = "[bold white on green]  SPEAKING  [/bold white on green]"
            else:
                status = "[bold white on dim red]  SILENT    [/bold white on dim red]"

            if recorder.transcription_worker.is_busy:
                transcribing_status = "[bold white on green]  TRANSCRIBING  [/bold white on green]"
            else:
                transcribing_status = "[bold white on dim red]  NOT TRANSCRIBING    [/bold white on dim red]"

            if recorder.realtime_transcription_worker.is_busy:
                realtime_transcribing_status = "[bold white on green]  REALTIME TRANSCRIBING  [/bold white on green]"
            else:
                realtime_transcribing_status = "[bold white on dim red]  NOT REALTIME TRANSCRIBING    [/bold white on dim red]"

            progress.update(task_id, completed=recorder.speech_confidence * 100)
            table.add_row(progress, status)

            # table.add_row(Text(f"size of buffer: {len(recorder.pre_audio_chunks_rolling_buffer)}"))
            table.add_row(Text(f"time taken to transcribe: {recorder.transcription_worker.time_taken_to_transcribe:.3f}"), transcribing_status)
            table.add_row(Text(f"time taken to realtime transcribe: {recorder.realtime_transcription_worker.time_taken_to_transcribe:.3f}"), realtime_transcribing_status)
            # table.add_row(Text(f"time taken to detect voice: {recorder.time_taken_to_detect_voice:.5f}"))
            # table.add_row(Text(f"time taken to record: {time_taken_to_record:.3f}"))
            table.add_row(Text(f"size of audio queue: {recorder.audio_queue.qsize()}"))
            table.add_row(Text(f"size of transcription queue (including currently transcribing): {recorder.transcription_worker.audio_to_transcribe_queue.qsize() + recorder.transcription_worker.is_busy}"))
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
    
