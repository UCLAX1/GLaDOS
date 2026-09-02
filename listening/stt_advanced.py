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


from silero_vad import load_silero_vad
import platform
import wave
import torch
import pyaudio
import audioop
import numpy as np
import torch
import time
from typing import Union, Callable, Optional
import queue
import collections
import threading
import multiprocessing as mp

import keyboard
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

def write_to_wav_file(file_name: str, audio_chunks: list[bytes], num_channels, sample_rate):
    with wave.open(file_name, mode="wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2) # 2 for 2-byte, 16-bit floats
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(audio_chunks))

def clear_queue(queue: Union[mp.Queue, queue.Queue]):
    while not queue.empty():
        queue.get()

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

def transcribed_text_handler(
    transcribed_text_queue: mp.Queue,
    stop_event: mp.Event,
    on_transcription_update_callback: Callable
):
    while not stop_event.is_set():
        try:
            transcribed_text = transcribed_text_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        threading.Thread(
            target=on_transcription_update_callback,
            args=(transcribed_text,),
            daemon=True,
        ).start()

def transcription_worker(
    audio_to_transcribe_queue: mp.Queue,
    transcriber_args,
    transcriber_loaded_event: mp.Event,
    transcribed_text_queue: mp.Queue,
    is_busy: mp.Value,
    time_taken_to_transcribe: mp.Value,
    stop_event: mp.Event,
    resume_event: mp.Event,
    skip_event: mp.Event,
    enable_early_transcription=False,
    should_keep_queue: mp.Queue=None, # only set for final transcription worker
    realtime_transcriber_is_busy: mp.Value=None, # only set for final transcription worker
    realtime_skip_event: mp.Event=None, # only set for final transcription worker
):

    transcriber = Transcriber(**transcriber_args)
    transcriber_loaded_event.set()

    try:
        while not stop_event.is_set():

            resume_event.wait()
            
            skip_event.clear()

            try:
                speech_chunks = audio_to_transcribe_queue.get(timeout=0.01)
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

            is_busy.value = True

            transcription_start_time = time.time()

            transcribed_text = transcriber.transcribe(speech_chunks)

            time_taken_to_transcribe.value = time.time() - transcription_start_time

            if skip_event.is_set():
                is_busy.value = False
                skip_event.clear()
                continue

            if not resume_event.is_set():
                is_busy.value = False
                continue

            if enable_early_transcription:
                # --- MORE EARLY TRANSCRIPTION LOGIC --- 
                if initially_awaiting_keep:
                    try:
                        should_keep = should_keep_queue.get(timeout=4)
                    except queue.Empty:
                        is_busy.value = False
                        raise Exception("FATAL ERROR: Timeout reached, should_keep queue remained empty after transcription completed")
                    if not should_keep:
                        is_busy.value = False
                        continue
                # --- END MORE EARLY TRANSCRIPTION LOGIC --- 

            transcribed_text_queue.put(transcribed_text)

            # stops the realtime transcriber so it doesn't finish after the final transcriber
            if (realtime_skip_event is not None) \
                and (realtime_transcriber_is_busy is not None) \
                and realtime_transcriber_is_busy.value:
                # print("skipping realtime worker")
                realtime_skip_event.set()

            is_busy.value = False

    except KeyboardInterrupt:
        pass
        # print("transcription worker: keyboard interrupt")

def audio_worker(
    audio_queue: mp.Queue,
    audio_process_loaded_event: mp.Event,
    sample_rate: int,
    frames_per_buffer: int,
    stop_event: mp.Event,
    resume_event: mp.Event,
):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    stream.start_stream()

    audio_process_loaded_event.set()

    try:
        while not stop_event.is_set():
            if not resume_event.is_set():
                stream.stop_stream()
                resume_event.wait()
                stream.start_stream()
            data: list[bytes] = stream.read(frames_per_buffer) # blocks
            audio_queue.put(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def vad_worker(
    vad_process_loaded_event: mp.Event,
    audio_queue,
    vad_device,
    pre_audio_chunks_rolling_buffer,
    speech_chunks,
    seconds_per_chunk,
    sample_rate,
    post_speech_silence_duration,
    speech_prob_threshold,
    min_speech_duration_for_transcription,
    enable_early_transcription,
    should_keep_queue,
    audio_to_final_transcribe_queue,
    enable_realtime_transcription,
    audio_to_realtime_transcribe_queue,
    final_transcription_worker_is_busy: mp.Value,
    realtime_transcription_worker_is_busy: mp.Value,
    speech_confidence: mp.Value,
    is_speaking: mp.Value,
    stop_event: mp.Event,
    resume_event: mp.Event,
    transcription_resume_event: mp.Event,
):

    vad_model = load_silero_vad()
    vad_model.to(vad_device)

    vad_process_loaded_event.set()

    vad_detects_speech = False
    vad_detects_speech_previous = False
    vad_detects_speech_start = False
    vad_detects_speech_stop = False
    time_vad_detects_speech_stop = 0.0
    time_submitted_final_transcription_request = 0.0

    try:
        while not stop_event.is_set():

            # reload vad and reset variables if recording is paused
            if not resume_event.is_set():
                vad_model = load_silero_vad()
                vad_model.to(vad_device)
                vad_detects_speech = False
                vad_detects_speech_previous = False
                vad_detects_speech_start = False
                vad_detects_speech_stop = False

            resume_event.wait()

            if (audio_queue.qsize() > 100):
                raise Exception("audio queue got too big, either hardware is too slow or I am bad at coding")

            try:
                audio_chunk = audio_queue.get(timeout=0.01)
                # audio_chunk = audio_queue.get_nowait()
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
            vad_detects_speech_start = vad_detects_speech and (not vad_detects_speech_previous)
            vad_detects_speech_stop = (not vad_detects_speech) and vad_detects_speech_previous
        
            if enable_early_transcription \
                and transcription_resume_event.is_set() \
                and vad_detects_speech_start \
                and is_speaking.value:
                # and final_transcription_worker_is_busy.value:
                # TODO: fix this ^^^

                should_keep_queue.put(False) # discard

            if vad_detects_speech_start and (not is_speaking.value):
                is_speaking.value = True
                time_vad_first_detects_speech = time.time()

                speech_chunks.extend(pre_audio_chunks_rolling_buffer)

                write_to_wav_file("pre-audio.wav", pre_audio_chunks_rolling_buffer, 1, sample_rate)

                pre_audio_chunks_rolling_buffer.clear()

            if vad_detects_speech_stop:
                # falling edge
                time_vad_detects_speech_stop = time.time()

            seconds_of_speech_stored = len(speech_chunks) * seconds_per_chunk
            
            # submit realtime transcription only when the realtime transcription worker is not already transcribing something
            if enable_realtime_transcription \
                and transcription_resume_event.is_set() \
                and vad_detects_speech \
                and seconds_of_speech_stored > 1.0 \
                and (not realtime_transcription_worker_is_busy.value) \
                and audio_to_realtime_transcribe_queue.empty():

                audio_to_realtime_transcribe_queue.put(speech_chunks)

            
            if enable_early_transcription:
                time_since_last_submitted_final_transcription_request = time.time() - time_submitted_final_transcription_request

                if vad_detects_speech_stop \
                    and transcription_resume_event.is_set() \
                    and time_since_last_submitted_final_transcription_request > 0.1 \
                    and seconds_of_speech_stored > min_speech_duration_for_transcription:

                    # print("submitting early transcription request")

                    audio_to_final_transcribe_queue.put(speech_chunks)
                    time_submitted_final_transcription_request = time.time()


            if (not vad_detects_speech) and is_speaking.value:

                elapsed_silence = time.time() - time_vad_detects_speech_stop

                if elapsed_silence > post_speech_silence_duration:
                    is_speaking.value = False
                    # print("submitting final transcription request")

                    if enable_early_transcription \
                        and transcription_resume_event.is_set():
                        # and final_transcription_worker_is_busy.value:

                        should_keep_queue.put(True) # keep

                    if (not enable_early_transcription) \
                        and transcription_resume_event.is_set():
                       
                        audio_to_final_transcribe_queue.put(speech_chunks)
                        time_submitted_final_transcription_request = time.time()

                    write_to_wav_file("microphone-results.wav", speech_chunks, 1, sample_rate)

                    speech_chunks.clear()

            vad_detects_speech_previous = vad_detects_speech

    except KeyboardInterrupt:
        pass
        # print("vad worker: keyboard interrupt")


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

        from faster_whisper import WhisperModel
        self.model = WhisperModel(self.model_type, device=self.device, compute_type=self.compute_type)

        # import nemo.collections.asr as nemo_asr
        # self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

        # pretty accurate and about 4-5 times faster
        # self.faster_whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")


    def transcribe(self, speech_chunks: list[bytes]) -> str:
        transcribed_text = ""

        speech_chunks_float32: np.ndarray = int16_bytes_list_to_normalized_float32_ndarray(speech_chunks)

        # return self.model.transcribe(speech_chunks_float32)[0].text


        # segments, info = self.faster_whisper_model.transcribe(speech_chunks_float32, language=self.language, condition_on_previous_text=True)
        segments, info = self.model.transcribe(speech_chunks_float32, language=self.language, condition_on_previous_text=False)

        for segment in segments:
            transcribed_text += segment.text
        return transcribed_text

        # # return recognizer.recognize_faster_whisper(audio_data, language="en")

class Recorder():

    # NUM_CHANNELS=1
    SAMPLE_RATE=16000
    CHUNK_SIZE = 512 # num frames per buffer

    CHUNKS_PER_SECOND = SAMPLE_RATE / CHUNK_SIZE
    SECONDS_PER_CHUNK = CHUNK_SIZE / SAMPLE_RATE

    def __init__(
        self,
        transcriber_device: str,
        vad_device: str,
        on_transcription_update_callback: Callable,
        on_realtime_transcription_update_callback: Callable=None,
        transcriber_model_type: str="distil-large-v3",
        realtime_transcriber_model_type: str="tiny",
        enable_realtime_transcription=False,
        enable_early_transcription=True,
        speech_prob_threshold=0.5,
        pre_audio_chunk_buffer_duration=0.3,
        post_speech_silence_duration=1.0,
        min_speech_duration_for_transcription=0.4,
        language: str=None,
    ):
        """Initialize recorder

        param: transcriber_device: "cuda" or "cpu"
        param: vad_device: "cuda" or "cpu"
        param: on_transcription_update_callback: 
            What to do when final transcription is done.
        param: on_realtime_transcription_update_callback: 
            What to do when realtime transcription is done.
        param: transcriber_model_type: 
            Model type of the final transcriber.
            Only if using faster-whisper, deprecated for nvidia-parakeet.
        param: realtime_transcriber_model_type: 
            Same as above but for the realtime transcriber
        param: enable_realtime_transcription: 
            Whether it provides transcriptions in real time while there is speech,
            or if it should only wait until a sentence is done to transcribe.
        param: enable_early_transcription: an optimization technique: 
            This makes it run the final transcription as early as it can, and overwrite it when new. 
            Basically it will be transcribing more, but the final transcriptions will come faster.
        param: speech_prob_threshold: 
            How confident the VAD has to be in order to declare audio as speech.
            Float from 0-1.
        param: pre_audio_chunk_buffer_duration:
            How many seconds of audio to store before speech is detected.
            This many seconds of audio is appended to the start of the speech buffer.
        param: post_speech_silence_duration:
            How many seconds of silent audio needs to be detected before
            the program declares there is no more speech.
        param: language:
            ex: "en"
            The language passed to the model.
            If None, then the model will try to guess the language.
        """


        self.speech_prob_threshold = speech_prob_threshold
        self.min_speech_duration_for_transcription = min_speech_duration_for_transcription
        self.pre_audio_chunk_buffer_duration = pre_audio_chunk_buffer_duration,

        self.pre_audio_chunk_buffer_size = int(pre_audio_chunk_buffer_duration * self.CHUNKS_PER_SECOND) # max num chunks

        self.enable_realtime_transcription = enable_realtime_transcription
        self.enable_early_transcription = enable_early_transcription
        self.language = language

        self.transcriber_device = transcriber_device
        self.vad_device = vad_device

        TranscriberThreadType: Union[threading.Thread, mp.Process] = threading.Thread
        if transcriber_device == "cpu":
            TranscriberThreadType = mp.Process

        # when we first detect speech, it's only after a few ms of speech is said, and we need to add that to the buffer
        self.pre_audio_chunks_rolling_buffer: collections.deque[list[bytes]] = collections.deque(
            maxlen=self.pre_audio_chunk_buffer_size
        ) 

        # chunks with speech with the pre-audio chunks appended to the front
        self.speech_chunks: list[bytes] = []

        self.audio_queue: mp.Queue[bytes] = mp.Queue()

        self.on_transcription_update_callback = on_transcription_update_callback
        self.on_realtime_transcription_update_callback = on_realtime_transcription_update_callback

        self.final_transcription_worker_is_busy = mp.Value('i', False)
        self.audio_to_final_transcribe_queue: mp.Queue[list[bytes]] = mp.Queue()
        self.should_keep_queue: mp.Queue[bool] = mp.Queue()

        self.realtime_transcription_worker_is_busy = mp.Value('i', False)
        self.audio_to_realtime_transcribe_queue: mp.Queue[list[bytes]] = mp.Queue()

        self.final_transcribed_text_queue: mp.Queue[str] = mp.Queue()
        self.realtime_transcribed_text_queue: mp.Queue[str] = mp.Queue()

        self.post_speech_silence_duration = post_speech_silence_duration # time to wait after speaking first not detected, in seconds

        self.recording_resume_event = mp.Event()
        self.transcription_resume_event = mp.Event()
        self.stop_event = mp.Event()

        self.is_speaking = mp.Value('i', 0)

        self.speech_confidence = mp.Value('d', 0.0)
        # self.boundary_detected: bool = False

        self.final_transcription_skip_event = mp.Event()
        self.realtime_transcription_skip_event = mp.Event()

        self.time_taken_to_final_transcribe = mp.Value('d', 0.0)

        final_transcriber_loaded_event = mp.Event()

        final_transcriber_args = dict(
            device=self.transcriber_device, 
            model_type=transcriber_model_type, 
            language=self.language
        )

        final_transcription_worker_args = dict(
            target=transcription_worker,
            args=(
                self.audio_to_final_transcribe_queue,
                final_transcriber_args,
                final_transcriber_loaded_event,
                self.final_transcribed_text_queue,
                self.final_transcription_worker_is_busy,
                self.time_taken_to_final_transcribe,
                self.stop_event,
                self.transcription_resume_event,
                self.final_transcription_skip_event,
                self.enable_early_transcription,
                self.should_keep_queue, 
                self.realtime_transcription_worker_is_busy,
                self.realtime_transcription_skip_event,
            ),
            daemon=True,
        )

        self.final_transcription_worker = TranscriberThreadType(**final_transcription_worker_args).start()


        self.time_taken_to_realtime_transcribe = mp.Value('d', 0.0)

        if self.enable_realtime_transcription:

            realtime_transcriber_loaded_event = mp.Event()

            realtime_transcriber_args = dict(
                device=self.transcriber_device,
                model_type=realtime_transcriber_model_type,
                language=self.language,
            )

            realtime_transcription_worker_args = dict(
                target=transcription_worker, # the transcription_worker function
                args=(
                    self.audio_to_realtime_transcribe_queue,
                    realtime_transcriber_args,
                    realtime_transcriber_loaded_event,
                    self.realtime_transcribed_text_queue,
                    self.realtime_transcription_worker_is_busy,
                    self.time_taken_to_realtime_transcribe,
                    self.stop_event,
                    self.transcription_resume_event,
                    self.realtime_transcription_skip_event,
                    False,
                    None,
                    None,
                    None
                ),
                daemon=True,
            )

            self.realtime_transcription_worker = TranscriberThreadType(**realtime_transcription_worker_args).start()

        self.final_text_handler = threading.Thread(
            target=transcribed_text_handler,
            args=(
                self.final_transcribed_text_queue,
                self.stop_event,
                self.on_transcription_update_callback,
            ),
            daemon=True,
        )
        self.final_text_handler.start()

        self.realtime_text_handler = threading.Thread(
            target=transcribed_text_handler,
            args=(
                self.realtime_transcribed_text_queue,
                self.stop_event,
                self.on_realtime_transcription_update_callback,
            ),
            daemon=True,
        )
        self.realtime_text_handler.start()

        audio_process_loaded_event = mp.Event()

        audio_process_args = dict(
            target=audio_worker,
            args=(
                self.audio_queue,
                audio_process_loaded_event,
                self.SAMPLE_RATE,
                self.CHUNK_SIZE,
                self.stop_event,
                self.recording_resume_event,
            ),
            daemon=True,
        )

        if platform.system() == 'Linux':
            self.audio_process = threading.Thread(**audio_process_args)
        else: 
            self.audio_process = mp.Process(**audio_process_args)

        self.audio_process.start()

        vad_process_loaded_event = mp.Event()

        self.vad_process = mp.Process(
            target=vad_worker, 
            args=(
                vad_process_loaded_event,
                self.audio_queue,
                self.vad_device,
                self.pre_audio_chunks_rolling_buffer,
                self.speech_chunks,
                self.SECONDS_PER_CHUNK,
                self.SAMPLE_RATE,
                self.post_speech_silence_duration,
                self.speech_prob_threshold,
                self.min_speech_duration_for_transcription,
                self.enable_early_transcription,
                self.should_keep_queue,
                self.audio_to_final_transcribe_queue,
                self.enable_realtime_transcription,
                self.audio_to_realtime_transcribe_queue,
                self.final_transcription_worker_is_busy,
                self.realtime_transcription_worker_is_busy,
                self.speech_confidence,
                self.is_speaking,
                self.stop_event,
                self.recording_resume_event,
                self.transcription_resume_event,
            ),
            daemon=True,
        )
        self.vad_process.start()

        audio_process_loaded_event.wait()
        print("audio process loaded")
        vad_process_loaded_event.wait()
        print("vad process loaded")
        final_transcriber_loaded_event.wait()
        print("final transcriber and worker loaded")
        if self.enable_realtime_transcription:
            realtime_transcriber_loaded_event.wait()
            print("realtime transcriber and worker loaded")

        print("finished init of recorder")
    
    def is_paused(self):
        return not self.recording_resume_event.is_set()

    def is_transcription_paused(self):
        return not self.transcription_resume_event.is_set()

    def start(self):
        self.resume()

    def resume(self):
        self.recording_resume_event.set()
        self.resume_transcription()

        # self.final_transcription_skip_event.clear()

        # if self.enable_realtime_transcription:
        #     self.realtime_transcription_skip_event.clear()
    
    def pause(self):

        self.recording_resume_event.clear()

        self.pause_transcription(clear_transcription_queue=False) # already clearing queues in _clear_buffers_and_values

        self._clear_buffers_and_values()

    
    def close(self):
        """
        Permanently closes the whole thing
        """
        self.stop_event.set()

        # resumes the workers so they can break out of their while loops
        self.recording_resume_event.set()
        self.transcription_resume_event.set()

        if self.final_transcription_worker is not None:
            self.final_transcription_worker.join()
        if self.enable_realtime_transcription and self.realtime_transcription_worker is not None:
            self.realtime_transcription_worker.join()
        if self.vad_process is not None:
            self.vad_process.join()

        self._clear_buffers_and_values()

        for q in [self.audio_queue, self.should_keep_queue, 
                        self.audio_to_final_transcribe_queue, 
                        self.audio_to_realtime_transcribe_queue]:
            q.cancel_join_thread() # Don't wait for internal buffer flush on exit
            q.close()


        # if self.final_transcription_worker is not None:
        #     self.final_transcription_worker.join()
        # if self.enable_realtime_transcription and self.realtime_transcription_worker is not None:
        #     self.realtime_transcription_worker.join()
        # if self.vad_process is not None:
        #     self.vad_process.join()
    
    def resume_transcription(self, clear_transcription_queue=True):
        if clear_transcription_queue:
            self._clear_transcription_queues()

        self.transcription_resume_event.set()

    def pause_transcription(self, clear_transcription_queue=True):
        self.transcription_resume_event.clear()

        if clear_transcription_queue:
            self._clear_transcription_queues()

        if self.final_transcription_worker_is_busy:
            self.final_transcription_skip_event.set()

        if self.enable_realtime_transcription and self.realtime_transcription_worker_is_busy:
            self.realtime_transcription_skip_event.set()


    def _clear_buffers_and_values(self):
        clear_queue(self.audio_queue)
        clear_queue(self.should_keep_queue)
        self._clear_transcription_queues()
        self.speech_chunks.clear()
        self.is_speaking.value = False
        self.speech_confidence.value = 0.0
        self.final_transcription_worker_is_busy.value = False
        self.realtime_transcription_worker_is_busy.value = False
    
    def _clear_transcription_queues(self):
        clear_queue(self.audio_to_final_transcribe_queue)
        if self.enable_realtime_transcription:
            clear_queue(self.audio_to_realtime_transcribe_queue)
        
    
    def __del__(self):
        if not self.stop_event.is_set():
            self.close()
    
if __name__ == '__main__':

    # transcribed_text = ""
    realtime_transcribed_text = ""

    def on_transcription_update(transcribed_text_output: str):
        global realtime_transcribed_text

        transcribed_text = preprocess_text(transcribed_text_output)
        full_sentences.append(transcribed_text)
        realtime_transcribed_text = ""
        # print(full_sentences)

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

    # check out the different model types here: https://github.com/SYSTRAN/faster-whisper/blob/ed9a06cd89a93e47838f564998a6c09b655d7f43/faster_whisper/transcribe.py#L640
    # distil-small is less accurate but around 3x faster
    # model_type = "distil-large-v3" if self.device == "cuda" else "distil-small.en"
    # model_type = "distil-small.en"
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
        # language="en",
    )
    recorder.start()

    start = time.time()

    print("Listening...")

    with Live(console=console, refresh_per_second=60) as live:
        try:
            while True:
                # if keyboard.is_pressed("p") and not recorder.is_paused():
                #     recorder.pause()

                # if keyboard.is_pressed("r") and recorder.is_paused():
                #     recorder.resume()

                # can't use rising edge detector since the keyboard library also detects auto-repeated key holds
                if keyboard.is_pressed("p") and not recorder.is_transcription_paused():
                    recorder.pause_transcription()

                if keyboard.is_pressed("r") and recorder.is_transcription_paused():
                    recorder.resume_transcription()

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

                if recorder.is_transcription_paused():
                    paused_status = "[bold white on green]  TRANSCRIPTION PAUSED  [/bold white on green]"
                else:
                    paused_status = "[bold white on dim red]  TRANSCRIPTION NOT PAUSED    [/bold white on dim red]"

                progress.update(task_id, completed=recorder.speech_confidence.value * 100)
                table.add_row(progress, status)

                # table.add_row(Text(f"size of buffer: {len(recorder.pre_audio_chunks_rolling_buffer)}"))
                table.add_row(Text(f"time taken to final transcribe: {recorder.time_taken_to_final_transcribe.value:.3f}"), transcribing_status)
                table.add_row(Text(f"time taken to realtime transcribe: {recorder.time_taken_to_realtime_transcribe.value:.3f}"), realtime_transcribing_status)
                # table.add_row(Text(f"time taken to detect voice: {recorder.time_taken_to_detect_voice:.5f}"))
                # table.add_row(Text(f"time taken to record: {time_taken_to_record:.3f}"))
                table.add_row(Text(f"size of audio queue: {recorder.audio_queue.qsize()}"), paused_status)
                table.add_row(Text(f"size of transcription queue (including currently transcribing): {recorder.audio_to_final_transcribe_queue.qsize() + recorder.final_transcription_worker_is_busy.value}"))
                table.add_row(Text(f"size of realtime transcription queue (including currently transcribing): {recorder.audio_to_realtime_transcribe_queue.qsize() + recorder.realtime_transcription_worker_is_busy.value}"))
                # table.add_row(Text(f"is paused: {recorder.is_paused()}"))
                table.add_row(Text(f"skip event: {recorder.final_transcription_skip_event.is_set()}"))
                # table.add_row(Text(f"time since detected speech stop: {silence_duration}"))
                table.add_row(transcribed_rich_text)
                panel = Panel(table, title="[bold]Live VAD Monitor[/bold]", border_style="blue")
                live.update(panel)

                # END DISPLAYING STUFF TO CONSOLE
        except KeyboardInterrupt:
            print("main: keyboard interrupt")
        finally:
            print("closing recorder...")
            recorder.close()
            print("done")
        
