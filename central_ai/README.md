# Central AI

Everything related to the GLaDOS central AI model lives here.

## Current recommendation

**Bonsai 8B via llama.cpp** is the top-performing model as of 2026-08-21:
- 12/12 (100%) persona benchmark
- 17/19 (89%) format/JSON benchmark
- 1.52s average latency
- 1.15 GB on disk (1-bit Q1_0 quantization)

See `benchmarking_tools/` for the full benchmark suite and leaderboard.

## Structure

- `benchmarking_tools/` — LLM evaluation suite (persona drift, JSON format, latency)