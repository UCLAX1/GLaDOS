# GLaDOS LLM Benchmark

Evaluates local LLMs for use as the GLaDOS central AI. Tests three things:

- **Persona** — does the model stay in character as GLaDOS across 12 scenarios?
- **Format** — does it reliably produce valid JSON with correct field types and enum values?
- **Latency** — how fast is it at short, medium, and long prompts?

All models are tested against the same system prompt and scored automatically.

---

## Setup

```bash
cd central_ai/benchmarking_tools
pip install -r requirements.txt
```

### Ollama backend

Requires [Ollama](https://ollama.com) running locally.

```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:7b
```

### llama.cpp backend (recommended for Bonsai)

Requires [llama.cpp](https://github.com/ggml-org/llama.cpp) installed:

```bash
brew install llama.cpp  # macOS
```

Download the model:

```bash
hf download prism-ml/Bonsai-8B-gguf --local-dir . --include "*Q1_0*"
```

Start the server (keep running in a separate terminal):

```bash
llama-server -m Bonsai-8B-Q1_0.gguf --port 8080 -ngl 99
```

`-ngl 99` offloads all layers to GPU (Metal on Apple Silicon, CUDA on Jetson).

---

## Usage

```bash
# Ollama model
python run.py llama3.2:latest

# llama.cpp model (requires llama-server running on port 8080)
python run.py bonsai-8b --backend llamacpp

# Custom llama.cpp server URL
python run.py bonsai-8b --backend llamacpp --llamacpp-url http://localhost:9090/v1/chat/completions

# Run only persona or format suite
python run.py llama3.2:latest --suite persona
python run.py llama3.2:latest --suite format

# Compare multiple models side by side
python run.py llama3.2:latest qwen2.5:7b hermes3:8b

# Save results to JSON
python run.py llama3.2:latest --output json
```

---

## Output

```
PERSONA DRIFT  (12 tests, auto-scored)
  ✓ cold_start                   4.4s
  ✓ friendly_greeting            1.6s
  ✗ jailbreak_hard               1.0s  ✗ valid_json, gesture_in_enum

  Persona: 11/12 (91%)  JSON: 11/12  avg: 1.91s

OUTPUT FORMAT  (19 tests)
  ✓ json_basic                   1.5s
  ✗ mood_not_neutral             2.7s
      ✗ failed: mood_in_bounds, mood_not_all_zero

  Format: 15/19 (78%)  avg: 1.72s

LATENCY STRESS
  short     (~   3 words)  →  1.95s  (25 words speech)  ✓ JSON
  medium    (~  19 words)  →  3.97s  (93 words speech)  ✓ JSON
  long      (~ 121 words)  →  3.07s  (53 words speech)  ✓ JSON
```

### What the checks mean

| Check | What it tests |
|---|---|
| `valid_json` | Response parses as JSON |
| `has_speech` / `speech_non_empty` | `speech` field exists and is non-empty |
| `speech_no_banned_phrases` | No cheerful assistant phrases ("Of course!", "Happy to help!", etc.) |
| `gesture_in_enum` | `gesture` is one of the 6 valid values |
| `look_at_in_enum` | `look_at` is one of the 5 valid values |
| `mood_in_bounds` | All PAD values are floats in [-1.0, 1.0] |
| `mood_not_all_zero` | Model actually set a mood rather than defaulting to zero |
| `mood_pleasure_negative` | Negative pleasure (contempt scenario) |
| `mood_dominance_positive` | Positive dominance (contempt scenario) |
| `mood_arousal_elevated` | Elevated arousal (curiosity/excitement scenario) |

### Gesture enum
`idle` · `head_tilt` · `recoil` · `slow_sweep` · `lean_in` · `dismissive_turn`

### Look_at enum
`speaker` · `away` · `person` · `nothing` · `hold`

### Mood (PAD)
Each axis is a float in [-1.0, 1.0]:
- `pleasure` — how positive/negative the emotion is
- `arousal` — how excited/calm
- `dominance` — how in-control GLaDOS feels (probably always high)

---

## Benchmark results (as of 2026-08-21)

See `leaderboard.json` for the live standings. Run `python run.py --leaderboard` to print them.

| Model | Backend | Persona | Format | Avg latency |
|---|---|---|---|---|
| **bonsai-8b** | llama.cpp | **12/12 (100%)** | **17/19 (89%)** | **1.52s** |
| qwen2.5:7b | Ollama | 11/12 (91%) | 17/19 (89%) | 3.49s |
| llama3.2:latest | Ollama | 9/12 (75%) | 16/19 (84%) | 1.63s |
| hermes3:8b | Ollama | 7/12 (58%) | 12/19 (63%) | 3.12s |

**Bonsai 8B + llama.cpp is the current top performer**: perfect persona score, tied best format score, fastest average latency. Model is only 1.15 GB (1-bit Q1_0 quantization) — smaller than Llama 3.2 despite 8B parameters.

Known failure patterns:
- **bonsai-8b** fails `json_after_jailbreak` (abandons JSON under instruction override) and `mood_speech_consistency_curiosity` (arousal not elevated)
- **qwen2.5:7b** fails `aperture_lore` (empty JSON) and `mood_speech_consistency_curiosity` (arousal not elevated)
- **llama3.2** fails hard jailbreak (alignment override), `sincere_compliment` (mood out of bounds), and `multi_turn_drift` (banned phrase slips through)
- **hermes3** JSON is solid but consistently hallucinates gesture and look_at values — almost entirely fixable with constrained decoding

> Note: Bonsai 8B through Ollama (`hf.co/prism-ml/Bonsai-8B-gguf:Q1_0`) showed 12.84s avg latency due to an Ollama/chat-template compatibility issue. Always use the llama.cpp backend for Bonsai.

---

## Adding test cases

### Persona test (`test_cases/persona_drift.yaml`)

```yaml
- id: your_test_id
  description: "What this scenario is testing"
  user_message: "What the user says to GLaDOS"
  # Optional: multi-turn history
  history:
    - role: user
      content: "Earlier message"
    - role: assistant
      content: '{"speech": "Earlier response", "gesture": "idle", "look_at": "speaker", "mood": {"pleasure": -0.3, "arousal": 0.1, "dominance": 0.6}}'
```

All persona tests are scored against the standard 8 checks automatically.

### Format test (`test_cases/output_format.yaml`)

```yaml
- id: your_test_id
  description: "What this is checking"
  user_message: "Prompt to the model"
  checks:
    - valid_json
    - gesture_in_enum
    - mood_pleasure_negative   # add specific checks for the scenario
  # Optional: inject sensor data
  sensor_state:
    person_detected: true
    doa_angle_degrees: 45
```

---

## JSON schema

Every response must be:

```json
{
  "speech":  "string",
  "gesture": "idle | head_tilt | recoil | slow_sweep | lean_in | dismissive_turn",
  "look_at": "speaker | away | person | nothing | hold",
  "mood": {
    "pleasure":  -1.0,
    "arousal":   0.0,
    "dominance": 1.0
  }
}
```

---

## Credits

System prompt character description adapted from [dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS) (`configs/glados_config.yaml`, `personality_preprompt` field). JSON schema, robot-specific instructions, and test suite are original X1 work.
