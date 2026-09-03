from recorder import Recorder

import keyboard
import time
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

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

if __name__ == '__main__':

    # transcribed_text = ""
    realtime_transcribed_text = ""

    def on_transcription_update(transcribed_text_output: str):
        global realtime_transcribed_text

        transcribed_text = preprocess_text(transcribed_text_output)
        full_sentences.append(transcribed_text)
        realtime_transcribed_text = ""

        # print(transcribed_text)

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
    # model_type = "distil-large-v3"
    # model_type = "large-v3"
    # model_type = "turbo"
    model_type = "parakeet"

    # be careful running the realtime model with parakeet since it takes up a lot of ram
    realtime_model_type = "distil-small.en"
    # realtime_model_type = "parakeet"

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
        