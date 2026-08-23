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
import multiprocessing as mp
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

def _transcription_worker(
    audio_to_transcribe_queue,
    transcriber,
    on_transcription_update_callback,
    is_busy,
    enable_early_transcription=False,
    should_keep_queue=None, # set if enable_early_transcription=True
):
    while True:
        try:
            speech_chunks = audio_to_transcribe_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if enable_early_transcription:
            # --- EARLY TRANSCRIPTION LOGIC --- 
            initially_awaiting_keep = should_keep_queue.empty()

            if not initially_awaiting_keep:
                should_keep = should_keep_queue.get_nowait()
                if not should_keep:
                    continue
            # --- END EARLY TRANSCRIPTION LOGIC --- 

        # transcription_start_time = time.time()
        is_busy.value = True

        transcribed_text = transcriber.transcribe(speech_chunks)

        # time_taken_to_transcribe = time.time() - transcription_start_time

        if enable_early_transcription:
            # --- MORE EARLY TRANSCRIPTION LOGIC --- 
            if initially_awaiting_keep:
                try:
                    should_keep = should_keep_queue.get(timeout=4)
                except queue.Empty:
                    raise Exception("FATAL ERROR: Timeout reached, should_keep queue remained empty after transcription completed")
                if not should_keep:
                    continue
            # --- END MORE EARLY TRANSCRIPTION LOGIC --- 

        threading.Thread(
            target=on_transcription_update_callback,
            args=(transcribed_text,),
            daemon=True,
        ).start()

        is_busy.value = False

def vad_worker(
    vad_process_loaded: mp.Value,
    audio_queue,
    vad_device,
    pre_audio_chunks_rolling_buffer,
    speech_chunks,
    seconds_per_chunk,
    sample_rate,
    post_speech_silence_duration,
    speech_prob_threshold,
    enable_early_transcription,
    should_keep_queue,
    audio_to_final_transcribe_queue,
    enable_realtime_transcription,
    audio_to_realtime_transcribe_queue,
    realtime_transcription_worker_is_busy: mp.Value,
    speech_confidence: mp.Value,
    is_speaking: mp.Value,
):
    vad_model = load_silero_vad()
    vad_model.to(vad_device)

    vad_process_loaded.value = True

    vad_detects_speech = False
    vad_detects_speech_previous = False
    vad_detects_speech_start = False
    vad_detects_speech_stop = False
    time_vad_detects_speech_stop = 0.0
    time_submitted_final_transcription_request = 0.0

    try:
        while True:

            if (audio_queue.qsize() > 100):
                raise Exception("audio queue got too big, either hardware is too slow or I am bad at coding")

            try:
                audio_chunk = audio_queue.get_nowait()
            except queue.Empty:
                continue

            if not is_speaking.value:
                pre_audio_chunks_rolling_buffer.append(audio_chunk)

            if is_speaking.value:
                speech_chunks.append(audio_chunk)

            vad_start_time = time.time()

            audio_chunk_float32: np.ndarray = int16_bytes_to_normalized_float32_ndarray(audio_chunk)
            audio_tensor = torch.from_numpy(audio_chunk_float32).to(vad_device)
            speech_confidence.value = vad_model(audio_tensor, sample_rate).item()
            # time.sleep(0.05) # <- if it takes too long to detect speech then the queue will grow infinitely
            # but this is less of a bug and more of a hardware limitation

            time_taken_to_detect_voice = time.time() - vad_start_time

            # rms = audioop.rms(audio_chunk, 2)

            # TODO: endpointing
            # https://arunbaby.com/speech-tech/0035-speech-boundary-detection/
            # see: "What is endpointing and why is a fixed timeout insufficient?"
            # right now, this code uses a fixed timeout of 1 second

            vad_detects_speech = speech_confidence.value > speech_prob_threshold
            vad_detects_speech_start = vad_detects_speech and not vad_detects_speech_previous
            vad_detects_speech_stop = not vad_detects_speech and vad_detects_speech_previous
        
            if enable_early_transcription:
                if vad_detects_speech_start and is_speaking.value:
                    should_keep_queue.put(False) # discard

            if vad_detects_speech_start and (not is_speaking.value):
                is_speaking.value = True
                time_vad_first_detects_speech = time.time()

                speech_chunks.extend(pre_audio_chunks_rolling_buffer)

                audio_data = sr.AudioData(b"".join(pre_audio_chunks_rolling_buffer), sample_rate=sample_rate, sample_width=2) # 2 for 2 byte, 16 bit ints
                with open("pre-audio.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())

                pre_audio_chunks_rolling_buffer.clear()

            if vad_detects_speech_stop:
                # falling edge
                time_vad_detects_speech_stop = time.time()

            seconds_of_speech_stored = len(speech_chunks) * seconds_per_chunk
            
            if enable_realtime_transcription:
                # submit realtime transcription only when the realtime transcription worker is not already transcribing something
                if vad_detects_speech and seconds_of_speech_stored > 1.0 and not realtime_transcription_worker_is_busy.value and audio_to_realtime_transcribe_queue.empty():
                    audio_to_realtime_transcribe_queue.put(speech_chunks)
            
            if enable_early_transcription:
                time_since_last_submitted_final_transcription_request = time.time() - time_submitted_final_transcription_request

                if vad_detects_speech_stop \
                    and time_since_last_submitted_final_transcription_request > 0.1:
                    # and seconds_of_speech_stored > 0.5:

                    # print("submitting early transcription request")
                    audio_to_final_transcribe_queue.put(speech_chunks)
                    time_submitted_final_transcription_request = time.time()

            if (not vad_detects_speech) and is_speaking.value:

                elapsed_silence = time.time() - time_vad_detects_speech_stop

                if elapsed_silence > post_speech_silence_duration:
                    is_speaking.value = False

                    if enable_early_transcription:
                        should_keep_queue.put(True) # keep

                    if not enable_early_transcription:
                        # print("submitting final transcription request")
                        audio_to_final_transcribe_queue.put(speech_chunks)
                        time_submitted_final_transcription_request = time.time()

                    # print("writing audio data to file")
                    audio_data = sr.AudioData(b"".join(speech_chunks), sample_rate=sample_rate, sample_width=2) # 2 for 2 byte, 16 bit ints
                    with open("microphone-results.wav", "wb") as f:
                        f.write(audio_data.get_wav_data())
                    # print("done writing audio data to file")

                    speech_chunks.clear()

            vad_detects_speech_previous = vad_detects_speech
    except KeyboardInterrupt:
        pass


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

class Recorder():

    NUM_CHANNELS=1
    SAMPLE_RATE=16000
    CHUNK_SIZE = 512 # num frames per buffer

    CHUNKS_PER_SECOND = SAMPLE_RATE / CHUNK_SIZE
    SECONDS_PER_CHUNK = CHUNK_SIZE / SAMPLE_RATE

    def __init__(
        self,
        transcriber_device: str,
        vad_device: str,
        on_transcription_update_callback,
        on_realtime_transcription_update_callback=None,
        transcriber_model_type="distil-large-v3",
        realtime_transcriber_model_type="tiny",
        enable_realtime_transcription=False,
        enable_early_transcription=True,
        speech_prob_threshold=0.5,
        pre_audio_chunk_buffer_duration=0.3,
        language=None
    ):
        self.speech_prob_threshold = speech_prob_threshold
        self.pre_audio_chunk_buffer_duration = pre_audio_chunk_buffer_duration,

        self.pre_audio_chunk_buffer_size = int(pre_audio_chunk_buffer_duration * self.CHUNKS_PER_SECOND) # max num chunks

        self.enable_realtime_transcription = enable_realtime_transcription
        self.enable_early_transcription = enable_early_transcription
        self.language = language

        self.transcriber_device = transcriber_device
        self.vad_device = vad_device

        # when we first detect speech, it's only after a few ms of speech is said, and we need to add that to the buffer
        self.pre_audio_chunks_rolling_buffer: collections.deque[list[bytes]] = collections.deque(
            maxlen=self.pre_audio_chunk_buffer_size
        ) 

        # chunks with speech with the pre-audio chunks appended to the front
        self.speech_chunks: list[bytes] = []

        self.audio_queue: mp.Queue[bytes] = mp.Queue()

        self.on_transcription_update_callback = on_transcription_update_callback
        self.on_realtime_transcription_update_callback = on_realtime_transcription_update_callback

        print("loading transcriber...")
        self.transcriber = Transcriber(device=self.transcriber_device, model_type=transcriber_model_type, language=self.language)
        print("done loading transcriber")

        self.audio_to_final_transcribe_queue: mp.Queue[list[bytes]] = mp.Queue()
        self.should_keep_queue: mp.Queue[bool] = mp.Queue()

        self.final_transcription_worker_is_busy = mp.Value('i', False)

        self.final_transcription_worker = threading.Thread(
            target=_transcription_worker,
            args=(
                self.audio_to_final_transcribe_queue,
                self.transcriber,
                self.on_transcription_update_callback,
                self.final_transcription_worker_is_busy,
                True,
                self.should_keep_queue, 
            ),
            daemon=True,
        ).start()

        self.realtime_transcription_worker_is_busy = mp.Value('i', 0)

        if self.enable_realtime_transcription:
            print("loading realtime transcriber...")
            self.realtime_transcriber = Transcriber(device=self.transcriber_device, model_type=realtime_transcriber_model_type, language=self.language)
            print("done loading realtime transcriber")

            self.audio_to_realtime_transcribe_queue: mp.Queue[list[bytes]] = mp.Queue()

            self.realtime_transcription_worker = threading.Thread(
                target=_transcription_worker,
                args=(
                    self.audio_to_realtime_transcribe_queue,
                    self.realtime_transcriber,
                    self.on_realtime_transcription_update_callback,
                    self.realtime_transcription_worker_is_busy,
                    False,
                    None,
                ),
                daemon=True,
            ).start()

        self.time_start = time.time()

        self.post_speech_silence_duration = 1.0 # time to wait after speaking first not detected, in seconds

        self.is_speaking = mp.Value('i', 0)

        self.speech_confidence = mp.Value('d', 0.0)
        # self.boundary_detected: bool = False

        self.vad_process_loaded = mp.Value('i', 0)

        self.vad_process = mp.Process(
            target=vad_worker, 
            args=(
                self.vad_process_loaded,
                self.audio_queue,
                self.vad_device,
                self.pre_audio_chunks_rolling_buffer,
                self.speech_chunks,
                self.SECONDS_PER_CHUNK,
                self.SAMPLE_RATE,
                self.post_speech_silence_duration,
                self.speech_prob_threshold,
                self.enable_early_transcription,
                self.should_keep_queue,
                self.audio_to_final_transcribe_queue,
                self.enable_realtime_transcription,
                self.audio_to_realtime_transcribe_queue,
                self.realtime_transcription_worker_is_busy,
                self.speech_confidence,
                self.is_speaking,
            ),
            daemon=True,
        )
        self.vad_process.start()

        while not self.vad_process_loaded.value:
            time.sleep(0.1)

        self.audio = pyaudio.PyAudio()
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

    def _on_new_audio_chunk_callback(self, audio_chunk: bytes, frame_count: int, time_info: dict, status: int) -> tuple:
        self.audio_queue.put(audio_chunk)
        return (None, pyaudio.paContinue)

if __name__ == '__main__':

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
                if recorder.is_speaking.value:
                    status = "[bold white on green]  SPEAKING  [/bold white on green]"
                else:
                    status = "[bold white on dim red]  SILENT    [/bold white on dim red]"

                if bool(recorder.final_transcription_worker_is_busy.value):
                    transcribing_status = "[bold white on green]  TRANSCRIBING  [/bold white on green]"
                else:
                    transcribing_status = "[bold white on dim red]  NOT TRANSCRIBING    [/bold white on dim red]"

                if recorder.realtime_transcription_worker_is_busy.value:
                    realtime_transcribing_status = "[bold white on green]  REALTIME TRANSCRIBING  [/bold white on green]"
                else:
                    realtime_transcribing_status = "[bold white on dim red]  NOT REALTIME TRANSCRIBING    [/bold white on dim red]"

                progress.update(task_id, completed=recorder.speech_confidence.value * 100)
                table.add_row(progress, status)

                # table.add_row(Text(f"size of buffer: {len(recorder.pre_audio_chunks_rolling_buffer)}"))
                # table.add_row(Text(f"time taken to transcribe: {recorder.transcription_worker.time_taken_to_transcribe:.3f}"), transcribing_status)
                table.add_row(Text(f""), transcribing_status)
                # table.add_row(Text(f"time taken to realtime transcribe: {recorder.realtime_transcription_worker.time_taken_to_transcribe:.3f}"), realtime_transcribing_status)
                # table.add_row(Text(f"time taken to detect voice: {recorder.time_taken_to_detect_voice:.5f}"))
                # table.add_row(Text(f"time taken to record: {time_taken_to_record:.3f}"))
                table.add_row(Text(f"size of audio queue: {recorder.audio_queue.qsize()}"))
                table.add_row(Text(f"size of transcription queue (including currently transcribing): {recorder.audio_to_final_transcribe_queue.qsize() + recorder.final_transcription_worker_is_busy.value}"))
                table.add_row(Text(f"size of realtime transcription queue (including currently transcribing): {recorder.audio_to_realtime_transcribe_queue.qsize() + recorder.realtime_transcription_worker_is_busy.value}"))
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
        
