# Smart speaker

A homemade, Yandex-Station-style voice assistant. Speech recognition, the default LLM and speech synthesis run locally; weather and Wikipedia lookups use the internet. Explicit requests can escalate to a remote LLM or an external agent harness.

The default character is a sarcastic Russian speaker woken by **«Эй, долбаёб»**. It answers **«Чё надо?»** to a bare wake phrase. Russian and English commands are supported.

## Parts

- **Voice** (`speaker/voice/`): microphone capture, wake/stop phrases, echo rejection, barge-in, follow-up listening and Silero TTS playback. The output device is opened only while something plays.
- **Speech gate** (`speaker/voice/gate.py`): an energy check and Silero VAD (~2 MB) skip silence, and Vosk small Russian (~50 MB) detects the wake phrase. Two seconds of pre-roll preserve the complete command. Our own playback (plus a 0.5 s tail) is ignored so answers can't wake the speaker.
- **ASR**: Nemotron 0.6B streaming through `nemo-speech`, on the Radeon iGPU via Vulkan. It activates after the small wake gate and stays active during an answer or ring. A Russian stream verifies the wake phrase and handles Russian commands; an optional auto-detect stream handles English once a conversation is open.
- **Brain** (`speaker/brain/`): LFM3-1.7B through `llama-server` on the Radeon iGPU, with timers, alarms, volume, arithmetic, weather and Wikipedia tools. Common commands bypass the LLM. Optional escalation streams a larger OpenAI-compatible API model or runs a real agent command. Personal memory uses EmbeddingGemma 2 semantic search through a second `llama-server`.
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
models/embeddinggemma-2-Q8_0.gguf
models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
models/silero/v5_6_ru.pt
models/silero/v3_en.pt
models/silero/vad.jit
models/vosk-model-small-ru-0.22/
```

LFM3 GGUF files are published at `LiquidAI/LFM3-1.7B-GGUF` on Hugging Face. LFM3 requires the llama.cpp build from this flake, not the older nixpkgs build. The embedding model is `embeddinggemma-2-Q8_0.gguf` from `ggml-org/embeddinggemma-2-GGUF`. The VAD file is `silero_vad.jit` from `snakers4/silero-vad`; the wake model is `vosk-model-small-ru-0.22` from `alphacephei.com/vosk/models`.

`alsa/speaker.conf` currently targets the Razer Kraken V3 X (`CARD=X`) and exposes shared `speaker_in`/`speaker_out` devices. Change the card there for standalone microphones or speakers, then re-enter `nix develop` (the dev shell copies the file). The capture period is 100 ms and the playback period 40 ms to keep idle CPU wakeups low. The current headset test setup plays the simulated user into the **right earcup**, near the microphone.

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

Retrieval uses **EmbeddingGemma 2** (text-only Q8_0 GGUF) served by `speaker embed` on the iGPU: about 18 ms per embedding, ~200 MB of memory, and no PyTorch in the brain. Before each LLM answer, memories closely related to the request (`recall_threshold`) are added to the prompt automatically, because LFM3 does not reliably decide to search on its own; broader questions such as «что ты обо мне помнишь?» use the `search_memory` tool. If the embedding server is down, answers still work without recall.

## Run

```sh
speaker up
```

For separate terminals, start in this order:

```sh
speaker llm
speaker embed
speaker asr
speaker brain
speaker voice
```

Use `speaker -v up` to include llama.cpp and ASR server logs. Stop `speaker up` with Ctrl-C; it stops all five processes.

Ask the running brain without using the microphone:

```sh
speaker ask "какая завтра погода в Питере?"
speaker ask "поставь таймер на пять минут" --speak
speaker ask "спроси большую модель: объясни квантовую запутанность"
speaker ask "запусти агента: проверь этот проект"
```

Use another configuration with `speaker --config /path/to/config.toml up` or `$SPEAKER_CONFIG`.

## Settings

```sh
speaker settings
```

Opens `config.toml` in a terminal editor: sections on the left, fields on the right, with each comment from the file shown above the keys it describes. Switches edit booleans, numbers are checked as you type (thresholds, volume and temperatures also against their ranges), lists take one item per line, and the ASR device and Russian voice are drop-downs. **Ctrl+S** saves, keeping the file's comments and layout, and lists the processes to restart; **Ctrl+R** reverts unsaved edits; **Ctrl+Q** quits (twice if there are unsaved changes).

## Voice behavior

- Say **«Эй, долбаёб, поставь таймер на пять минут»**, or say the wake phrase alone and wait for the acknowledgement.
- During speech or a ringing timer, any speech interrupts and becomes the next command; the wake phrase is not needed. A short **«стоп»**, **«хватит»** or another configured stop word just stops the response.
- Ordinary answers return to idle. A follow-up without the wake phrase is accepted only when the speaker explicitly asked you a question. Requests have no conversation history; only timers, alarms and volume persist.
- Start with **«спроси большую модель»** to use `[remote_llm]`, or **«запусти агента»** to run `[harness]`. English triggers are also configured.
- Volume commands use levels 0–10. Timers and alarms persist in `state/brain.json`; personal memories persist in `state/memory.sqlite3`.
- Music playback and smart-home control are not implemented.

Tune wake matching, stop words, microphone gain, output channels, TTS voices, endpoint silence, the 4096-token generation ceiling, escalation backends and the character prompt in `config.toml`, by hand or with `speaker settings`.
