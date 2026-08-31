#!/usr/bin/env python3
"""
train_kokoro.py  —  GLaDOS Kokoro-82M fine-tuning pipeline
============================================================

Strategy
--------
Kokoro-82M is a StyleTTS2 model (iSTFTNet decoder) trained by hexgrad.
Fine-tuning uses StyleTTS2's train_finetune_accelerate.py with a hybrid
pretrained checkpoint that combines:

  • Kokoro-82M weights (bert/PLBERT, bert_encoder, text_encoder,
                         predictor, decoder/iSTFTNet)           ← language + acoustics
  • StyleTTS2-LibriTTS weights (style_encoder, predictor_encoder) ← voice style encoding

This lets us:
  - Keep Kokoro's superior text+prosody model
  - Use LibriTTS's trained StyleEncoder to encode GLaDOS reference audio
  - Fine-tune on the robotic dataset so the model specialises to the voice

Dataset
-------
  Input:  ~/voice_training/dataset/robotic/  (LJSpeech format, 22050 Hz)
  Output: ~/voice_training/checkpoints/kokoro_robotic/

Usage
-----
  python train_kokoro.py          # full pipeline: prep + train
  python train_kokoro.py --prep   # data prep only (no training)
  python train_kokoro.py --train  # training only (skip prep)
"""

import argparse, csv, json, math, os, shutil, subprocess, sys, yaml
from pathlib import Path

import numpy as np
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
HOME        = Path.home()
DATASET_DIR = HOME / "voice_training/dataset/robotic"
OUTPUT_DIR  = HOME / "voice_training/checkpoints/kokoro_robotic"
WORK        = HOME / "voice_training/kokoro_work"
STYLETTS2   = WORK / "StyleTTS2"
KOKORO_DIR  = WORK / "kokoro"
LIBRITTS    = WORK / "libritts/Models/LibriTTS"
DATA_DIR    = WORK / "data"
HYBRID_PTH  = WORK / "kokoro_hybrid_pretrained.pth"
FT_CONFIG   = WORK / "kokoro_ft.yml"

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
SAMPLE_RATE     = 24000        # Kokoro's native sample rate
BATCH_SIZE      = 8            # reduce to 4 if OOM
MAX_LEN_FRAMES  = 400          # ~5 sec at 24kHz/hop300; lower if OOM
EPOCHS          = 50
VAL_FRACTION    = 0.05         # 5% validation split
LEARNING_RATE   = 1e-4
BERT_LR         = 1e-5
FT_LR           = 1e-4
# Skip expensive losses by pushing their start epoch past training end
DIFF_EPOCH      = 9999         # skip style diffusion training
JOINT_EPOCH     = 9999         # skip SLM (WavLM) adversarial training

# ─── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[kokoro] {msg}", flush=True)


def run(cmd: str, cwd: Path | None = None) -> None:
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


# ─── Step 1: Resample dataset from 22050 Hz → 24000 Hz ───────────────────────

def prepare_data() -> None:
    """Resample wavs and build train/val list files."""
    import soundfile as sf
    import librosa

    log("=== Step 1: Preparing dataset ===")

    wavs_in  = DATASET_DIR / "wavs"
    meta_in  = DATASET_DIR / "metadata.csv"
    wavs_out = DATA_DIR / "wavs_24k"
    wavs_out.mkdir(parents=True, exist_ok=True)

    if not meta_in.exists():
        raise FileNotFoundError(f"metadata.csv not found at {meta_in}")
    if not wavs_in.exists():
        raise FileNotFoundError(f"wavs/ directory not found at {wavs_in}")

    # Parse LJSpeech metadata: filename|transcript
    entries: list[tuple[str, str]] = []
    with open(meta_in, encoding="utf-8-sig") as f:
        for row in csv.reader(f, delimiter="|"):
            if len(row) < 2:
                continue
            name, text = row[0].strip(), row[1].strip()
            if not text or len(text) < 3:
                continue
            src = wavs_in / f"{name}.wav"
            if not src.exists():
                continue
            entries.append((name, text))

    log(f"Found {len(entries)} valid entries in metadata.csv")

    # Resample to 24kHz (skip if already done)
    resampled = 0
    skipped   = 0
    for name, _text in entries:
        src  = wavs_in / f"{name}.wav"
        dst  = wavs_out / f"{name}.wav"
        if dst.exists():
            skipped += 1
            continue
        audio, sr = librosa.load(str(src), sr=None, mono=True)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        sf.write(str(dst), audio, SAMPLE_RATE, subtype="PCM_16")
        resampled += 1

    log(f"Resampled {resampled} files, skipped {skipped} (already done).")

    # Only keep entries that have resampled wavs
    valid = [(n, t) for n, t in entries if (wavs_out / f"{n}.wav").exists()]
    log(f"Valid after resampling: {len(valid)}")

    # Train/val split (deterministic)
    np.random.seed(42)
    idx = np.random.permutation(len(valid))
    n_val   = max(1, math.floor(len(valid) * VAL_FRACTION))
    val_idx = set(idx[:n_val].tolist())
    train_lines = []
    val_lines   = []

    for i, (name, text) in enumerate(valid):
        wav_path = str(wavs_out / f"{name}.wav")
        # StyleTTS2 finetune format: audio_path|text|speaker
        line = f"{wav_path}|{text}|glados"
        if i in val_idx:
            val_lines.append(line)
        else:
            train_lines.append(line)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train_list.txt").write_text("\n".join(train_lines) + "\n")
    (DATA_DIR / "val_list.txt").write_text("\n".join(val_lines) + "\n")
    log(f"Written train_list.txt ({len(train_lines)} clips) + val_list.txt ({len(val_lines)} clips)")

    # OOD texts (out-of-distribution — used for SLM adversarial training)
    # We set joint_epoch=9999 so this file is never actually used, but
    # train_finetune.py requires it to exist.
    ood_texts = [t for _, t in valid[:200]]
    ood_path  = DATA_DIR / "ood_texts.txt"
    ood_path.write_text("\n".join(f"{t}|glados" for t in ood_texts) + "\n")
    log(f"Written ood_texts.txt ({len(ood_texts)} entries)")


# ─── Step 2: Build hybrid pretrained checkpoint ───────────────────────────────

def build_hybrid_checkpoint() -> None:
    """
    Merge Kokoro-82M and StyleTTS2-LibriTTS weights into a single checkpoint
    in the format StyleTTS2's load_checkpoint() expects.

    Kokoro provides: bert, bert_encoder, text_encoder, predictor, decoder
    LibriTTS provides: style_encoder, predictor_encoder
    (discriminators and diffusion are randomly init'd during training)
    """
    if HYBRID_PTH.exists():
        log(f"Hybrid checkpoint already exists at {HYBRID_PTH} — skipping.")
        return

    log("=== Step 2: Building hybrid pretrained checkpoint ===")

    kokoro_path  = KOKORO_DIR / "kokoro-v1_0.pth"
    libritts_path = LIBRITTS / "epochs_2nd_00020.pth"

    if not kokoro_path.exists():
        raise FileNotFoundError(f"Kokoro weights not found: {kokoro_path}\n"
                                 "Run setup_kokoro.sh first.")
    if not libritts_path.exists():
        raise FileNotFoundError(f"LibriTTS checkpoint not found: {libritts_path}\n"
                                 "Run setup_kokoro.sh first.")

    log(f"Loading Kokoro weights from {kokoro_path}...")
    kokoro_state = torch.load(str(kokoro_path), weights_only=True, map_location="cpu")
    # Format: {bert: state_dict, bert_encoder: state_dict, ...}

    log(f"Loading LibriTTS weights from {libritts_path}...")
    libritts_raw  = torch.load(str(libritts_path), map_location="cpu")
    libritts_state = libritts_raw.get("net", libritts_raw)

    log(f"Kokoro keys:   {sorted(kokoro_state.keys())}")
    log(f"LibriTTS keys: {sorted(k for k in libritts_state.keys() if not k.startswith('mpd') and not k.startswith('msd'))}")

    # Components to take from each source
    merged: dict = {}

    # From Kokoro: language + acoustic components
    for key in ("bert", "bert_encoder", "text_encoder", "predictor", "decoder"):
        if key in kokoro_state:
            merged[key] = kokoro_state[key]
            log(f"  [Kokoro]   {key}")
        else:
            log(f"  WARNING: {key} not found in Kokoro checkpoint!")

    # From LibriTTS: style encoding components
    # StyleEncoder and ProsodyPredictor encoder have compatible architectures
    # (same dim_in=64, style_dim=128, max_conv_dim=512 in both models)
    for key in ("style_encoder", "predictor_encoder"):
        if key in libritts_state:
            merged[key] = libritts_state[key]
            log(f"  [LibriTTS] {key}")
        else:
            log(f"  WARNING: {key} not found in LibriTTS checkpoint — will init from random.")

    # Optionally copy diffusion from LibriTTS (we won't train it but it needs to load)
    if "diffusion" in libritts_state:
        # Diffusion architecture differs when multispeaker=True (StyleTransformer1d)
        # LibriTTS is also multispeaker, so architecture should match.
        merged["diffusion"] = libritts_state["diffusion"]
        log("  [LibriTTS] diffusion")

    # Save in StyleTTS2 format: {net: {...}, epoch: 0, iters: 0}
    output = {"net": merged, "epoch": 0, "iters": 0}
    torch.save(output, str(HYBRID_PTH))
    log(f"Hybrid checkpoint saved to {HYBRID_PTH}")


# ─── Step 3: Extract PLBERT and save in StyleTTS2 Utils/PLBERT/ format ────────

def setup_plbert() -> None:
    """
    StyleTTS2's train_finetune_accelerate.py loads PLBERT from Utils/PLBERT/.
    It expects:
      Utils/PLBERT/config.yml  — PLBERT architecture config
      Utils/PLBERT/step_1000000.t7  — pretrained PLBERT weights

    Since Kokoro ships its own PLBERT weights (in bert key), we extract and
    re-save them in the format StyleTTS2 expects.
    """
    plbert_dir = STYLETTS2 / "Utils" / "PLBERT"
    plbert_dir.mkdir(parents=True, exist_ok=True)

    t7_path  = plbert_dir / "step_1000000.t7"
    yml_path = plbert_dir / "config.yml"

    if t7_path.exists() and yml_path.exists():
        log("PLBERT already set up — skipping.")
        return

    log("=== Step 3: Setting up PLBERT from Kokoro weights ===")

    # Load Kokoro's bert weights
    kokoro_path  = KOKORO_DIR / "kokoro-v1_0.pth"
    kokoro_state = torch.load(str(kokoro_path), weights_only=True, map_location="cpu")

    if "bert" not in kokoro_state:
        raise RuntimeError("'bert' key not found in Kokoro checkpoint.")

    bert_state = kokoro_state["bert"]

    # StyleTTS2's load_plbert() (Utils/PLBERT/util.py) expects:
    #   checkpoint = torch.load(step_XXXXXX.t7)
    #   state_dict = checkpoint['net']
    #   for k, v in state_dict.items():
    #       name = k[7:]            # strip 'module.'
    #       if name.startswith('encoder.'): name = name[8:]  # strip 'encoder.'
    #   del new_state_dict["embeddings.position_ids"]
    #   bert.load_state_dict(new_state_dict, strict=False)
    #
    # So we need keys in the form 'module.encoder.<albert_key>'.
    # Kokoro's bert state dict has plain AlbertModel keys (no prefix).
    # We add 'module.encoder.' so the loader strips them back correctly.

    from collections import OrderedDict
    transformed: dict = OrderedDict()
    for k, v in bert_state.items():
        transformed[f"module.encoder.{k}"] = v

    torch.save({"net": transformed}, str(t7_path))
    log(f"Saved PLBERT weights ({len(transformed)} keys) to {t7_path}")

    # Write PLBERT config.yml.  The loader does:
    #   plbert_config = yaml.safe_load(open(config_path))
    #   albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    # So we need a 'model_params' top-level key.
    kokoro_config_path = KOKORO_DIR / "config.json"
    kokoro_cfg = json.loads(kokoro_config_path.read_text())
    plbert_raw  = kokoro_cfg["plbert"]

    plbert_yml = {
        "model_params": {
            "hidden_size":              plbert_raw["hidden_size"],
            "num_attention_heads":      plbert_raw["num_attention_heads"],
            "intermediate_size":        plbert_raw["intermediate_size"],
            "max_position_embeddings":  plbert_raw["max_position_embeddings"],
            "num_hidden_layers":        plbert_raw["num_hidden_layers"],
            "attention_probs_dropout_prob": plbert_raw.get("dropout", 0.1),
            "hidden_dropout_prob":          plbert_raw.get("dropout", 0.1),
            # Kokoro vocab size for phoneme tokens
            "vocab_size":               kokoro_cfg["n_token"],
        }
    }
    yml_path.write_text(yaml.dump(plbert_yml, allow_unicode=True))
    log(f"Saved PLBERT config to {yml_path}")


# ─── Step 4: Write Kokoro fine-tuning config ──────────────────────────────────

def write_ft_config() -> None:
    """Write the StyleTTS2 fine-tuning YAML config for Kokoro's architecture."""
    if FT_CONFIG.exists():
        log(f"Fine-tune config already exists at {FT_CONFIG} — skipping.")
        return

    log("=== Step 4: Writing fine-tune config ===")

    cfg = {
        # ── I/O ──────────────────────────────────────────────────────────────
        "log_dir":                   str(OUTPUT_DIR),
        "save_freq":                 5,
        "log_interval":              10,
        "device":                    "cuda",
        "epochs":                    EPOCHS,
        "batch_size":                BATCH_SIZE,
        "max_len":                   MAX_LEN_FRAMES,

        # ── Checkpoint loading ────────────────────────────────────────────────
        "pretrained_model":          str(HYBRID_PTH),
        "second_stage_load_pretrained": True,
        "load_only_params":          True,

        # ── Utility models ────────────────────────────────────────────────────
        "F0_path":   "Utils/JDC/bst.t7",
        "ASR_config": "Utils/ASR/config.yml",
        "ASR_path":  "Utils/ASR/epoch_00080.pth",
        "PLBERT_dir": "Utils/PLBERT/",

        # ── Data ──────────────────────────────────────────────────────────────
        "data_params": {
            "train_data": str(DATA_DIR / "train_list.txt"),
            "val_data":   str(DATA_DIR / "val_list.txt"),
            "root_path":  "",         # paths in list files are absolute
            "OOD_data":   str(DATA_DIR / "ood_texts.txt"),
            "min_length": 50,
        },

        # ── Preprocessing ─────────────────────────────────────────────────────
        "preprocess_params": {
            "sr": SAMPLE_RATE,
            "spect_params": {
                "n_fft":       2048,
                "win_length":  1200,
                "hop_length":  300,
            },
        },

        # ── Model architecture (must match Kokoro-82M exactly) ────────────────
        "model_params": {
            "multispeaker":          True,
            "dim_in":                64,
            "hidden_dim":            512,
            "max_conv_dim":          512,
            "n_layer":               3,
            "n_mels":                80,
            "n_token":               178,
            "max_dur":               50,
            "style_dim":             128,
            "text_encoder_kernel_size": 5,
            "dropout":               0.2,

            # Kokoro uses iSTFTNet decoder
            "decoder": {
                "type":                    "istftnet",
                "resblock_kernel_sizes":   [3, 7, 11],
                "upsample_rates":          [10, 6],
                "upsample_initial_channel": 512,
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_kernel_sizes":   [20, 12],
                "gen_istft_n_fft":         20,
                "gen_istft_hop_size":      5,
            },

            # SLM discriminator (WavLM) — not used until joint_epoch
            "slm": {
                "model":           "microsoft/wavlm-base-plus",
                "sr":              16000,
                "hidden":          768,
                "nlayers":         13,
                "initial_channel": 64,
            },

            # Style diffusion — not used until diff_epoch
            "diffusion": {
                "embedding_mask_proba": 0.1,
                "transformer": {
                    "num_layers":    3,
                    "num_heads":     8,
                    "head_features": 64,
                    "multiplier":    2,
                },
                "dist": {
                    "sigma_data":        0.2,
                    "estimate_sigma_data": True,
                    "mean":              -3.0,
                    "std":               1.0,
                },
            },
        },

        # ── Loss weights ──────────────────────────────────────────────────────
        "loss_params": {
            "lambda_mel":  5.0,
            "lambda_gen":  1.0,
            "lambda_slm":  1.0,
            "lambda_mono": 1.0,
            "lambda_s2s":  1.0,
            "lambda_F0":   1.0,
            "lambda_norm": 1.0,
            "lambda_dur":  1.0,
            "lambda_ce":   20.0,
            "lambda_sty":  1.0,
            "lambda_diff": 1.0,
            # Push expensive losses past total epochs so they never fire:
            "diff_epoch":  DIFF_EPOCH,
            "joint_epoch": JOINT_EPOCH,
        },

        # ── Optimiser ─────────────────────────────────────────────────────────
        "optimizer_params": {
            "lr":      LEARNING_RATE,
            "bert_lr": BERT_LR,
            "ft_lr":   FT_LR,
        },

        # ── SLM adversarial params (not used, joint_epoch > epochs) ───────────
        "slmadv_params": {
            "min_len":          400,
            "max_len":          500,
            "batch_percentage": 0.5,
            "iter":             10,
            "thresh":           5,
            "scale":            0.01,
            "sig":              1.5,
        },
    }

    FT_CONFIG.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
    log(f"Fine-tune config written to {FT_CONFIG}")


# ─── Step 5: Patch StyleTTS2's PLBERT loader to handle our saved format ───────

def patch_plbert_loader() -> None:
    """
    StyleTTS2's PLBERT loader in train_finetune_accelerate.py uses a specific
    loading pattern.  We monkey-patch Utils/PLBERT/__init__.py (if it exists)
    to handle both the standard format and our extracted format.

    Also patches the main training script to use absolute paths for data files
    since our root_path is empty.
    """
    plbert_init = STYLETTS2 / "Utils" / "PLBERT" / "__init__.py"

    if not plbert_init.exists():
        # Create a stub __init__ so the package is importable
        plbert_init.write_text("")
        log(f"Created {plbert_init}")

    # Check if train_finetune_accelerate.py exists
    ft_script = STYLETTS2 / "train_finetune_accelerate.py"
    if not ft_script.exists():
        # Fall back to train_finetune.py
        ft_script_alt = STYLETTS2 / "train_finetune.py"
        if ft_script_alt.exists():
            log("train_finetune_accelerate.py not found; will use train_finetune.py")
        else:
            raise FileNotFoundError(
                "Neither train_finetune_accelerate.py nor train_finetune.py found in StyleTTS2/")

    log("PLBERT loader check complete.")


# ─── Step 6: Launch training ──────────────────────────────────────────────────

def launch_training() -> None:
    log("=== Step 6: Launching training ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer accelerate launcher (fp16, single GPU)
    ft_accelerate = STYLETTS2 / "train_finetune_accelerate.py"
    ft_regular    = STYLETTS2 / "train_finetune.py"

    if ft_accelerate.exists():
        cmd = (
            f"accelerate launch --mixed_precision=fp16 --num_processes=1 "
            f"train_finetune_accelerate.py "
            f"--config_path {FT_CONFIG}"
        )
        log("Using accelerate launcher with fp16.")
    else:
        cmd = f"python train_finetune.py --config_path {FT_CONFIG}"
        log("Using standard Python launcher.")

    log(f"Working directory: {STYLETTS2}")
    log(f"Command: {cmd}")
    log("Training output → stdout (which nohup redirects to /tmp/train_kokoro.log)")
    log(f"Checkpoints → {OUTPUT_DIR}")
    log("=" * 60)

    # exec into the training process (replaces this script's process)
    os.chdir(str(STYLETTS2))
    os.execvp("bash", ["bash", "-c", cmd])


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Kokoro-82M fine-tuning pipeline")
    parser.add_argument("--prep",  action="store_true", help="Data prep only")
    parser.add_argument("--train", action="store_true", help="Training only (skip prep)")
    args = parser.parse_args()

    do_prep  = not args.train   # default: do everything
    do_train = not args.prep

    log("GLaDOS Kokoro-82M fine-tuning pipeline")
    log(f"  Dataset:  {DATASET_DIR}")
    log(f"  Output:   {OUTPUT_DIR}")
    log(f"  Work dir: {WORK}")
    log(f"  Epochs:   {EPOCHS}  |  Batch: {BATCH_SIZE}  |  SR: {SAMPLE_RATE} Hz")
    log("")

    if do_prep:
        prepare_data()
        build_hybrid_checkpoint()
        setup_plbert()
        write_ft_config()
        patch_plbert_loader()

    if do_train:
        launch_training()   # This call does not return — it exec's into training


if __name__ == "__main__":
    main()
