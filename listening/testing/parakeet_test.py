# disable logging
from nemo.utils.nemo_logging import Logger

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)


# def blank_log(*args, **kwargs):
#     pass

# from nemo.utils import logging as logger
# logger.debug = blank_log
# logger.info = blank_log
# logger.warning = blank_log
# logger.error = blank_log
# logger.critical = blank_log

import nemo.collections.asr as nemo_asr
import time

start = time.time()

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

print(f"time taken to load model: {time.time() - start}")

while True:
    start = time.time()
    # output = asr_model.transcribe(["audio.mp3"])
    output = asr_model.transcribe(["../microphone-results.wav"], verbose=False)

    print(f"time taken to transcribe: {time.time() - start}")

    print(output[0].text)

# import time


# class Transcriber():

#     def __init__(
#         self,
#         device: str,
#         model_type: str,
#         language=None,
#     ):
#         """
#         device: either "cuda" or "cpu"
#         """

#         self.device = device

#         self.compute_type = "float16" if self.device == "cuda" else "int8"

#         self.model_type = model_type

#         self.language = language

#         # disabling the stupid logging
#         from nemo.utils.nemo_logging import Logger
#         nemo_logger = Logger()
#         nemo_logger.remove_stream_handlers()
#         import logging
#         logging.getLogger('nemo_logger').setLevel(logging.ERROR)
#         logging.disable(logging.CRITICAL)

#         import nemo.collections.asr as nemo_asr
#         self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

#         # import nemo.collections.asr as nemo_asr
#         # self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

#         # pretty accurate and about 4-5 times faster
#         # self.faster_whisper_model = WhisperModel("distil-small.en", device="cuda", compute_type="float16")
#     def transcribe(self) -> str:
#         transcribed_text = ""

#         print("starting transcription...")
#         transcribed_text = self.model.transcribe(["../audio-to-transcribe.wav"], verbose=False)[0].text
#         print("done transcribing")
#         return transcribed_text

# start = time.time()

# transcriber = Transcriber("cuda", "parakeet")

# print(f"time taken to load transcriber: {time.time() - start}")

# start = time.time()

# print(transcriber.transcribe())

# print(f"time taken to transcribe: {time.time() - start}")