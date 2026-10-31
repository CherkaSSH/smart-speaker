# Smart speaker

A homemade, Yandex-Station-style voice assistant. Speech recognition, the default LLM and speech synthesis run locally; weather and Wikipedia lookups use the internet. Explicit requests can escalate to a remote LLM or an external agent harness.

The default character is a sarcastic Russian speaker woken by **«Эй, долбаёб»**. It answers **«Чё надо?»** to a bare wake phrase. Russian and English commands are supported.

## Parts

- **Voice** (`speaker/voice/`): microphone capture, wake/stop phrases, echo rejection, barge-in, follow-up listening and Silero TTS playback.
- **Speech gate** (`speaker/voice/gate.py`): Silero VAD (~2 MB) skips silence and Vosk small Russian (~50 MB) detects the wake phrase. Two seconds of pre-roll preserve the complete command.
- **ASR**: Nemotron 0.6B streaming through `nemo-speech`, on CPU. It activates after the small wake gate and stays active during an answer or ring. A Russian stream handles wake verification and Russian commands; an optional auto-detect stream handles English.
- **Brain** (`speaker/brain/`): LFM3-1.7B through `llama-server` on the Radeon iGPU, with timers, alarms, volume, arithmetic, weather and Wikipedia tools. Common commands bypass the LLM. Optional escalation streams a larger OpenAI-compatible API model or runs a real agent command. Personal memory uses local EmbeddingGemma 2 semantic search.
- **Configuration**: one `config.toml` shared by every process.

## Setup

Requires Nix with flakes and an ALSA audio device.

```sh
nix develop
uv sync
source .venv/bin/activate
```

The model files are not downloaded by the flake. Put these files under `models/` (or change the paths in `config.toml`):

```text
models/LFM3-1.7B-QAD-Q4_0.gguf
models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
models/silero/v5_6_ru.pt
models/silero/v3_en.pt
models/silero/vad.jit
models/vosk-model-small-ru-0.22/
```

LFM3 GGUF files are published at `LiquidAI/LFM3-1.7B-GGUF` on Hugging Face. LFM3 requires the llama.cpp build from this flake, not the older nixpkgs build. The VAD file is `silero_vad.jit` from `snakers4/silero-vad`; the wake model is `vosk-model-small-ru-0.22` from `alphacephei.com/vosk/models`.

`alsa/speaker.conf` currently targets the Razer Kraken V3 X (`CARD=X`) and exposes shared `speaker_in`/`speaker_out` devices. Change the card there for standalone microphones or speakers. The current headset test setup plays the simulated user into the **right earcup**, near the microphone.

### Optional escalation

Both modes are disabled by default. With remote mode enabled, LFM3 can also invoke `ask_big_model` automatically for requests beyond its abilities; routine requests remain local.

For a larger API model, set `enabled = true` in `[remote_llm]`, configure its OpenAI-compatible URL and model, and export the key:

```sh
export REMOTE_LLM_API_KEY=...
speaker ask "Prove that there are infinitely many primes" --big
```

For an actual agent harness, set `enabled = true` and its argv `command` in `[harness]`. Every `%p` argument is replaced with the prompt; no shell is involved:

```toml
command = "pi -p %p"
```

```sh
speaker ask "Inspect this repository and summarize its architecture" --harness
```

The harness supplies its own model authentication and tools. API and harness modes are mutually exclusive.

### Personal memory

`save_memory` and `search_memory` let LFM3 persist durable facts and retrieve them when relevant. For example, say **«запомни, что я люблю овсяное молоко»**, then later ask what milk you prefer. Text and normalized 768-dimensional embeddings are stored in `state/memory.sqlite3`.

Retrieval uses **EmbeddingGemma 2** in its 270M-parameter text-only mode, locally on CPU with float32 precision (it loads in about 3 seconds when the brain starts; each search takes about 50 ms). The first start downloads the model from Hugging Face. Before each LLM answer, memories closely related to the request (`recall_threshold`) are added to the prompt automatically, because LFM3 does not reliably decide to search on its own; broader questions such as «что ты обо мне помнишь?» use the `search_memory` tool.

## Run

```sh
speaker up
```

For separate terminals, start in this order:

```sh
speaker llm
speaker asr
speaker brain
speaker voice
```

Use `speaker -v up` to include llama.cpp and ASR server logs. Stop `speaker up` with Ctrl-C; it stops all four processes.

Ask the running brain without using the microphone:

```sh
speaker ask "какая завтра погода в Питере?"
speaker ask "поставь таймер на пять минут" --speak
speaker ask "спроси большую модель: объясни квантовую запутанность"
speaker ask "запусти агента: проверь этот проект"
```

Use another configuration with `speaker --config /path/to/config.toml up` or `$SPEAKER_CONFIG`.

## Voice behavior

- Say **«Эй, долбаёб, поставь таймер на пять минут»**, or say the wake phrase alone and wait for the acknowledgement.
- During speech or a ringing timer, any speech interrupts and becomes the next command; the wake phrase is not needed. A short **«стоп»**, **«хватит»** or another configured stop word just stops the response.
- Ordinary answers return to idle. A follow-up without the wake phrase is accepted only when the speaker explicitly asked you a question. Requests have no conversation history; only timers, alarms and volume persist.
- Start with **«спроси большую модель»** to use `[remote_llm]`, or **«запусти агента»** to run `[harness]`. English triggers are also configured.
- Volume commands use levels 0–10. Timers and alarms persist in `state/brain.json`; personal memories persist in `state/memory.sqlite3`.
- Music playback and smart-home control are not implemented.

Tune wake matching, stop words, microphone gain, output channels, TTS voices, endpoint silence, the 4096-token generation ceiling, escalation backends and the character prompt in `config.toml`.
