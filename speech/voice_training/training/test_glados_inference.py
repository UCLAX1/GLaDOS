"""
Quick inference + latency/memory check for the fine-tuned GLaDOS Kokoro voice.
"""

import sys
import time
from pathlib import Path

import soundfile as sf
import torch

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "kokoro"))
from kokoro import KModel, KPipeline

CHECKPOINT = _repo_root / "StyleTTS2" / "logs" / "glados" / "epoch_2nd_00004.pth"
VOICEPACK = _repo_root / "voices" / "glados.pt"
CONVERTED = _repo_root / "voices" / "glados_kmodel.pth"
OUT_DIR = _repo_root / "test_output"

TEST_SENTENCES = [
    "The Enrichment Center regrets to inform you that this next test is impossible.",
    "Please note that we have added a consequence for failure.",
    "Any decision you make in the next ten seconds will be your last.",
    "This is your fault. I told you not to trust me, and you did it anyway.",
]


def convert_checkpoint():
    ckpt = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        return {("module." + k if not k.startswith("module.") else k): v for k, v in state_dict.items()}

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        kokoro_weights[key] = ensure_module_prefix(net[key])
    CONVERTED.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(CONVERTED))
    print(f"Converted checkpoint -> {CONVERTED}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not CONVERTED.exists():
        convert_checkpoint()

    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=str(CONVERTED))
    kmodel = kmodel.to(device).eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=kmodel)
    load_time = time.time() - t0
    print(f"Model load time: {load_time:.2f}s")

    voice = torch.load(str(VOICEPACK), map_location="cpu", weights_only=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_audio_s = 0.0
    total_gen_s = 0.0
    for i, text in enumerate(TEST_SENTENCES):
        t0 = time.time()
        generator = pipeline(text, voice=voice, speed=1)
        all_audio = []
        for gs, ps, audio in generator:
            all_audio.append(audio)
        audio = torch.cat(all_audio, dim=0) if len(all_audio) > 1 else all_audio[0]
        gen_time = time.time() - t0
        audio_dur = len(audio) / 24000
        total_audio_s += audio_dur
        total_gen_s += gen_time
        rtf = gen_time / audio_dur
        print(f"[{i+1}] '{text[:50]}...' -> {audio_dur:.2f}s audio in {gen_time:.3f}s (RTF={rtf:.3f})")
        sf.write(str(OUT_DIR / f"glados_test_{i}.wav"), audio.numpy(), 24000)

    print(f"\nTotal: {total_audio_s:.2f}s audio generated in {total_gen_s:.3f}s")
    print(f"Overall RTF: {total_gen_s/total_audio_s:.3f} (lower is faster than real-time)")

    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak GPU memory during inference: {peak_mem:.1f} MB")

    n_params = sum(p.numel() for p in kmodel.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")


if __name__ == "__main__":
    main()
