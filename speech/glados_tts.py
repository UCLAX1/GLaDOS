#!/usr/bin/env python3
"""
GLaDOS TTS — Kokoro KModel inference.

Usage:
    python3 glados_tts.py "The cake is a lie."
    python3 glados_tts.py "The cake is a lie." --out cake.wav --speed 0.95

Dependencies:
    pip install kokoro "misaki[en]" inflect soundfile huggingface_hub
    brew install espeak-ng          # Mac
    sudo apt-get install espeak-ng  # Jetson / Linux

Model files are downloaded automatically from HuggingFace on first run
(yifanfang/glados-kokoro) and cached in ~/.cache/huggingface/.
"""

import argparse
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning,  module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import soundfile as sf
import torch
from kokoro import KModel, KPipeline

# ── paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config_kokoro.json"   # ships in repo, always present
SAMPLE_RATE = 24000
HF_REPO     = "yifanfang/glados-kokoro"  # hugging face of my trained weights

# ── device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ── HuggingFace auto-download ───────────────────────────────────────────────
def _ensure_model_files() -> tuple[Path, Path]:
    """
    Return (model_path, voicepack_path), downloading from HF if not cached.
    Checks next to the script first (for offline / custom setups), then falls
    back to the HF cache.
    """
    local_model    = HERE / "glados_kmodel.pth"
    local_voicepack = HERE / "glados.pt"

    if local_model.exists() and local_voicepack.exists():
        return local_model, local_voicepack

    # Fall back to HuggingFace cache (downloads on first run, instant thereafter)
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    def _hf(filename):
        cached = try_to_load_from_cache(repo_id=HF_REPO, filename=filename, repo_type="model")
        if cached and cached != "NOT_FOUND_ON_HUB":
            return Path(cached)
        print(f"Downloading {filename} from {HF_REPO} …")
        return Path(hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="model"))
    return _hf("glados_kmodel.pth"), _hf("glados.pt")


# ── text normalisation ─────────────────────────────────────────────────────
def normalize_numbers(text: str) -> str:
    """Spell out standalone integers so the model doesn't mispronounce them."""
    import inflect
    p = inflect.engine()
    return re.sub(r"\b\d+\b", lambda m: p.number_to_words(m.group()), text)


# ── weight_norm format conversion ─────────────────────────────────────────
def _convert_weight_norm(state_dict: dict) -> dict:
    """
    Convert new-style weight_norm keys (parametrizations.weight.original0/1)
    to the old-style keys (weight_g / weight_v) that KModel's modules expect.

    PyTorch ≥ 2.x saves weight-normed layers with the parametrize API;
    KModel's own init uses weight_g / weight_v. We use tensor size to
    distinguish: weight_g is the small per-channel norm, weight_v is full.
    """
    out = {}
    groups: dict[str, dict] = {}
    for k, v in state_dict.items():
        if ".parametrizations.weight.original" in k:
            base = k[: k.index(".parametrizations.weight.original")]
            idx  = k[-1]
            groups.setdefault(base, {})[idx] = v
        else:
            out[k] = v
    for base, params in groups.items():
        t0, t1 = params["0"], params["1"]
        if t0.numel() <= t1.numel():
            out[base + ".weight_g"] = t0
            out[base + ".weight_v"] = t1
        else:
            out[base + ".weight_g"] = t1
            out[base + ".weight_v"] = t0
    return out


# ── model ──────────────────────────────────────────────────────────────────
class GladosTTS:
    def __init__(self, device: str | None = None):
        self.device = device or DEVICE
        model_path, voicepack_path = _ensure_model_files()

        self.kmodel = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(CONFIG_PATH),
        ).to(self.device).eval()

        ck = torch.load(str(model_path), map_location="cpu", weights_only=False)
        for comp_name, comp_state in ck.items():
            clean = {k.replace("module.", "", 1): v for k, v in comp_state.items()}
            clean = _convert_weight_norm(clean)
            submod = getattr(self.kmodel, comp_name, None)
            if submod is None:
                continue
            submod.load_state_dict(clean, strict=False)

        # torch.compile gives ~20-40% speedup after the first inference
        # (disabled on MPS — not yet fully supported)
        if self.device != "mps":
            try:
                self.kmodel = torch.compile(self.kmodel)
            except Exception:
                pass  # compile is optional; silently skip if unsupported

        self.pipeline = KPipeline(
            lang_code="a",
            repo_id="hexgrad/Kokoro-82M",
            model=self.kmodel,
        )
        self.voice = torch.load(str(voicepack_path), map_location="cpu", weights_only=True)

        # Warmup: run one silent forward pass so torch.compile finishes JIT
        # compilation before the first real utterance hits.
        with torch.no_grad():
            _ = self.speak("Initializing.", speed=1.0)

    def speak(self, text: str, speed: float = 1.0) -> torch.Tensor:
        text = normalize_numbers(text)
        chunks = [
            audio
            for _, _, audio in self.pipeline(text, voice=self.voice, speed=speed)
        ]
        audio = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return audio.clamp(-1.0, 1.0)


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GLaDOS TTS")
    parser.add_argument("text")
    parser.add_argument("--out",   default="out.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    tts   = GladosTTS()
    audio = tts.speak(args.text, speed=args.speed)
    sf.write(args.out, audio.numpy(), SAMPLE_RATE)
    print(f"Wrote {args.out}  ({len(audio)/SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
